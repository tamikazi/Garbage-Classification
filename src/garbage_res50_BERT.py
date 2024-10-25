import os
import re
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

import torchvision
from torchvision import transforms, models

from transformers import BertTokenizer, BertModel
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

import wandb

# ---------------------------
# Helper Functions and Classes
# ---------------------------

def extract_data_from_folders(base_dir):
    """Extract image paths, text descriptions, and labels from folders."""
    data = []
    # Traverse through each subfolder
    for label_folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, label_folder)
        # Check if it's a directory
        if os.path.isdir(folder_path):
            # Loop through each image file in the subfolder
            for filename in os.listdir(folder_path):
                if filename.endswith(('.jpg', '.png', '.jpeg')):  # Filter image files
                    image_path = os.path.join(folder_path, filename)
                    # Extract text from filename (remove file extension)
                    text_description = os.path.splitext(filename)[0]
                    # Append image path, text, and label to the data list
                    text = text_description.replace('_', ' ')
                    text_without_digits = re.sub(r'\d+', '', text).strip().lower()
                    data.append({
                        'image_path': image_path,
                        'text_description': text_without_digits,
                        'label': label_folder  # The subfolder name represents the label
                    })
    # Convert to DataFrame for easy manipulation
    return pd.DataFrame(data)

class GarbageDataset(Dataset):
    """Custom Dataset for Garbage Classification."""

    def __init__(self, dataframe, image_transform=None, max_len=32,
                 tokenizer=None, class_to_idx=None):
        self.dataframe = dataframe
        self.image_transform = image_transform
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.class_to_idx = class_to_idx

        self._token_cache = {}

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Get image path, text description, and label from the dataframe
        img_path = self.dataframe.iloc[idx]['image_path']
        text_desc = self.dataframe.iloc[idx]['text_description']
        label = self.dataframe.iloc[idx]['label']

        # Load and preprocess the image
        image = Image.open(img_path).convert("RGB")

        if self.image_transform:
            image = self.image_transform(image)

        # Tokenize the text description using caching
        if text_desc in self._token_cache:
            text_inputs = self._token_cache[text_desc]
        else:
            text_inputs = self.tokenizer(
                text_desc,
                add_special_tokens=True,
                padding='max_length',
                truncation=True,
                max_length=self.max_len,
                return_token_type_ids=False,
                return_tensors="pt"
            )
            self._token_cache[text_desc] = text_inputs

        # Convert string label to numeric label using the class mapping
        numeric_label = self.class_to_idx[label]

        # Return the image, text input, and numeric label
        return {
            'image': image,
            'input_ids': text_inputs['input_ids'].squeeze(0),
            'attention_mask': text_inputs['attention_mask'].squeeze(0),
            'label': torch.tensor(numeric_label, dtype=torch.long),
            'text_description': text_desc  # For logging misclassified examples
        }

# Define the image model using ResNet-50
class ImageModel(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(ImageModel, self).__init__()
        self.model = models.resnet50(pretrained=True)
        # Remove the last classification layer
        self.model.fc = nn.Identity()
        # Feature extractor to output a feature vector
        self.feature_extractor = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        x = self.model(x)  # x will have shape (batch_size, 2048)
        x = self.feature_extractor(x)  # x will have shape (batch_size, 512)
        return x

# Define the text model using BERT
class TextModel(nn.Module):
    def __init__(self, dropout_rate=0.5, pretrained_model_name='bert-base-uncased'):
        super(TextModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.feature_extractor = nn.Linear(self.bert.config.hidden_size, 512)

    def forward(self, input_ids, attention_mask):
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token_embedding = outputs.last_hidden_state[:, 0, :]  # Extract the CLS token
        pooled_output = self.dropout(cls_token_embedding)
        text_features = self.feature_extractor(pooled_output)
        return text_features

# GarbageClassifier model
class GarbageClassifier(nn.Module):
    def __init__(self, num_classes=4, dropout_rate=0.5):
        super(GarbageClassifier, self).__init__()
        # Image feature extraction with ResNet-50
        self.image_model = ImageModel(dropout_rate=dropout_rate)
        # Text model
        self.text_model = TextModel(dropout_rate=dropout_rate)
        # Fusion and classification layers
        self.fusion = nn.Sequential(
            nn.Linear(512 + 512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, input_ids, attention_mask):
        # Get image features
        image_features = self.image_model(image)  # Shape: (batch_size, 512)
        # Get text features (logits)
        text_features = self.text_model(input_ids, attention_mask)  # Shape: (batch_size, num_classes)
        # Combined features
        combined_features = torch.cat((image_features, text_features), dim=1)  # Shape: (batch_size, 512 + num_classes)
        # Combined prediction
        combined_output = self.fusion(combined_features)      # Shape: (batch_size, num_classes)
        return combined_output

# UnNormalize class
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(3,1,1)
        self.std = torch.tensor(std).view(3,1,1)

    def __call__(self, tensor):
        tensor = tensor * self.std + self.mean
        return torch.clamp(tensor, 0, 1)

# ---------------------------
# Training, Validation, and Testing Functions
# ---------------------------

def train_one_epoch(model, trainloader, criterion, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    running_loss_combined = 0.0
    running_corrects_combined = 0

    for i, batch in enumerate(trainloader):
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        # Forward pass
        combined_outputs = model(
            images, input_ids, attention_mask
        )

        # Compute loss
        loss_combined = criterion(combined_outputs, labels)

        # Backward pass and optimization
        loss_combined.backward()
        optimizer.step()

        # Update running loss
        running_loss_combined += loss_combined.item() * images.size(0)

        # Predictions
        _, preds_combined = torch.max(combined_outputs, 1)

        # Update running corrects
        running_corrects_combined += torch.sum(preds_combined == labels.data)

    # Compute epoch loss and accuracy
    epoch_loss_combined = running_loss_combined / len(trainloader.dataset)
    epoch_acc_combined = running_corrects_combined.double() / len(trainloader.dataset)

    return epoch_loss_combined, epoch_acc_combined

def validate(model, valloader, criterion, device):
    """Validate the model."""
    model.eval()
    val_running_loss_combined = 0.0
    val_running_corrects_combined = 0

    all_labels = []
    all_preds_combined = []

    with torch.no_grad():
        for batch in valloader:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # Forward pass
            combined_outputs = model(
                images, input_ids, attention_mask
            )

            # Compute loss
            loss_combined = criterion(combined_outputs, labels)

            # Update running loss
            val_running_loss_combined += loss_combined.item() * images.size(0)

            # Predictions
            _, preds_combined = torch.max(combined_outputs, 1)

            # Update running corrects
            val_running_corrects_combined += torch.sum(
                preds_combined == labels.data
            )

            # Collect labels and predictions
            all_labels.extend(labels.cpu().numpy())
            all_preds_combined.extend(preds_combined.cpu().numpy())

    # Compute validation loss and accuracy
    val_loss_combined = val_running_loss_combined / len(valloader.dataset)
    val_acc_combined = val_running_corrects_combined.double() / len(valloader.dataset)

    # Compute additional metrics
    precision = precision_score(
        all_labels, all_preds_combined, average='weighted', zero_division=0
    )
    recall = recall_score(
        all_labels, all_preds_combined, average='weighted', zero_division=0
    )
    f1 = f1_score(
        all_labels, all_preds_combined, average='weighted', zero_division=0
    )

    return val_loss_combined, val_acc_combined, precision, recall, f1

def test(model, testloader, device, idx_to_class):
    """Test the model and log misclassified examples."""
    model.eval()
    test_running_corrects_combined = 0

    all_labels = []
    all_preds_combined = []

    misclassified_table = wandb.Table(
        columns=["Image", "Predicted Label", "True Label", "Text Description"]
    )

    # Instantiate the UnNormalize transform
    unnormalize = UnNormalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])

    with torch.no_grad():
        for batch in testloader:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            text_descriptions = batch['text_description']

            # Forward pass
            combined_outputs = model(
                images, input_ids, attention_mask
            )

            # Predictions
            _, preds_combined = torch.max(combined_outputs, 1)

            test_running_corrects_combined += torch.sum(
                preds_combined == labels.data
            )

            # Collect labels and predictions
            all_labels.extend(labels.cpu().numpy())
            all_preds_combined.extend(preds_combined.cpu().numpy())

            # Log misclassified examples
            for img, pred_label, true_label, text in zip(
                images.cpu(), preds_combined.cpu(), labels.cpu(), text_descriptions
            ):
                if pred_label != true_label:
                    # Unnormalize the image
                    unnormalized_img = unnormalize(img)
                    # Convert the tensor to a PIL image
                    img_pil = transforms.ToPILImage()(unnormalized_img)
                    misclassified_table.add_data(
                        wandb.Image(img_pil),
                        idx_to_class[pred_label.item()],
                        idx_to_class[true_label.item()],  
                        text
                    )

    # Compute test accuracy
    test_acc_combined = test_running_corrects_combined.double() / len(testloader.dataset)

    # Compute additional metrics
    precision = precision_score(
        all_labels, all_preds_combined, average='weighted', zero_division=0
    )
    recall = recall_score(
        all_labels, all_preds_combined, average='weighted', zero_division=0
    )
    f1 = f1_score(
        all_labels, all_preds_combined, average='weighted', zero_division=0
    )

    # Confusion matrix
    conf_mat = confusion_matrix(all_labels, all_preds_combined)

    return test_acc_combined, precision, recall, f1, misclassified_table, conf_mat

# ---------------------------
# Training Function
# ---------------------------

def train():
    """Training function to be used with wandb sweeps."""
    # Initialize a new wandb run
    with wandb.init() as run:
        config = wandb.config

        # Initialize device
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {device}')

        # Define dataset directories
        trainset_dir = '/work/TALC/enel645_2024f/garbage_data/CVPR_2024_dataset_Train'
        valset_dir = '/work/TALC/enel645_2024f/garbage_data/CVPR_2024_dataset_Val'
        testset_dir = '/work/TALC/enel645_2024f/garbage_data/CVPR_2024_dataset_Test'

        # Define classes and map them to indices
        class_names = ['Green', 'Blue', 'Black', 'TTR']
        class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
        idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}

        # Extract the data
        trainset_df = extract_data_from_folders(trainset_dir)
        valset_df = extract_data_from_folders(valset_dir)
        testset_df = extract_data_from_folders(testset_dir)

        # Initialize the BERT tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Image transformations
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Create datasets
        trainset = GarbageDataset(
            trainset_df,
            image_transform=transform_train,
            class_to_idx=class_to_idx,
            tokenizer=tokenizer,
            max_len=32
        )
        valset = GarbageDataset(
            valset_df,
            image_transform=transform_test,
            class_to_idx=class_to_idx,
            tokenizer=tokenizer,
            max_len=32
        )    
        testset = GarbageDataset(
            testset_df,
            image_transform=transform_test,
            class_to_idx=class_to_idx,
            tokenizer=tokenizer,
            max_len=32
        )

        # DataLoaders
        trainloader = DataLoader(
            trainset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=2
        )
        valloader = DataLoader(
            valset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=2
        )
        testloader = DataLoader(
            testset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=2
        )

        # Compute class weights
        class_labels = np.unique(trainset_df['label'])
        class_weights = compute_class_weight(
            'balanced',
            classes=class_labels,
            y=trainset_df['label']
        )
        # Create a mapping from class labels to weights
        class_weights_dict = {label: weight for label, weight in zip(class_labels, class_weights)}
        # Map the class weights to the class indices
        class_weights_list = [class_weights_dict[class_name] for class_name in class_names]
        class_weights_tensor = torch.tensor(class_weights_list, dtype=torch.float).to(device)
        # Define the loss function with class weights
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

        # Initialize the model
        model = GarbageClassifier(num_classes=len(class_names), dropout_rate=config.dropout_rate).to(device)

        # Freeze and unfreeze layers as needed
        # Image model
        for param in model.image_model.model.parameters():
            param.requires_grad = False
        # for param in model.image_model.model.layer3.parameters():
        #     param.requires_grad = True
        # for param in model.image_model.model.layer4.parameters():
        #     param.requires_grad = True
        for param in model.image_model.feature_extractor.parameters():
            param.requires_grad = True

        # Text model
        for param in model.text_model.bert.parameters():
            param.requires_grad = True
        for param in model.text_model.feature_extractor.parameters():
            param.requires_grad = True

        # Collect trainable parameters
        trainable_params = [
            # Image feature extractor
            # {params: model.image_model.model.layer3.parameters(), 'lr': config.learning_rate},
            # {'params': model.image_model.model.layer4.parameters(), 'lr': config.learning_rate},
            {'params': model.image_model.feature_extractor.parameters(), 'lr': config.learning_rate},
            # Text model
            {'params': model.text_model.bert.parameters(), 'lr': config.learning_rate * 0.1},
            {'params': model.text_model.feature_extractor.parameters(), 'lr': config.learning_rate},
            # Fusion and classifier layers
            {'params': model.fusion.parameters(), 'lr': config.learning_rate},
        ]

        # Initialize the optimizer
        optimizer = optim.Adam(trainable_params, lr=config.learning_rate, weight_decay=1e-5)

        # Watch the model with wandb
        wandb.watch(model, log="all", log_freq=100)

        best_val_acc = 0.0  # Variable to track the best validation accuracy

        for epoch in range(10):  # Fixed epochs to 10
            print(f'Epoch {epoch + 1}/10')
            print('-' * 10)

            # Training phase
            epoch_loss_combined, epoch_acc_combined = train_one_epoch(
                model, trainloader, criterion, optimizer, device
            )

            print(f'Training Combined Loss: {epoch_loss_combined:.4f} '
                  f'Acc: {epoch_acc_combined:.4f}')

            wandb.log({
                "Epoch": epoch + 1,
                "Training Combined Loss": epoch_loss_combined,
                "Training Combined Accuracy": epoch_acc_combined.item(),
            })

            # Validation phase
            val_loss_combined, val_acc_combined, precision, recall, f1 = validate(
                model, valloader, criterion, device
            )

            print(f'Validation Combined Loss: {val_loss_combined:.4f} '
                  f'Acc: {val_acc_combined:.4f}')
            print(f'Validation Precision: {precision:.4f}, '
                  f'Recall: {recall:.4f}, F1-score: {f1:.4f}')

            wandb.log({
                "Validation Combined Loss": val_loss_combined,
                "Validation Combined Accuracy": val_acc_combined.item(),
                "Validation Precision": precision,
                "Validation Recall": recall,
                "Validation F1-score": f1
            })

            # Save the best model based on combined validation accuracy
            if val_acc_combined > best_val_acc:
                best_val_acc = val_acc_combined
                print(f"New best model found! Saving model with validation "
                      f"accuracy: {best_val_acc:.4f}")
                torch.save(model.state_dict(), 'best_model.pth')
                # Log model checkpoint as artifact
                artifact = wandb.Artifact('best_model', type='model')
                artifact.add_file('best_model.pth')
                wandb.log_artifact(artifact)

        # Load the best model and evaluate on the test set
        model.load_state_dict(torch.load('best_model.pth'))

        test_acc_combined, precision, recall, f1, misclassified_table, conf_mat = test(
            model, testloader, device, idx_to_class
        )

        print(f'Test Combined Accuracy: {test_acc_combined:.4f}')
        print(f'Test Precision: {precision:.4f}, '
              f'Recall: {recall:.4f}, F1-score: {f1:.4f}')

        wandb.log({
            "Test Combined Accuracy": test_acc_combined.item(),
            "Test Precision": precision,
            "Test Recall": recall,
            "Test F1-score": f1,
            "Misclassified Examples": misclassified_table
        })

        # Confusion matrix for test set
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(
            conf_mat,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Test Confusion Matrix')
        wandb.log({"Test Confusion Matrix": wandb.Image(fig)})
        plt.close(fig)

        # Finish the wandb run
        wandb.finish()

# ---------------------------
# Sweep Configuration and Initialization
# ---------------------------

def main():
    """Main function to define and run the wandb sweep."""
    # Define sweep configuration
    sweep_config = {
        'method': 'bayes',  # Options: 'grid', 'random', 'bayes'
        'metric': {
            'name': 'Validation Combined Accuracy',
            'goal': 'maximize'   
        },
        'parameters': {
            'batch_size': {
                'values': [16, 32, 64, 128, 256]
            },
            'learning_rate': {
                'min': 0.00001,
                'max': 0.0001
            },
            'dropout_rate': {
                'min': 0.2,
                'max': 0.8
            }
            # Add more hyperparameters to tune if needed
        }
    }

    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project='garbage-collection')

    # Run the sweep agent
    wandb.agent(sweep_id, function=train)

if __name__ == '__main__':
    main()
