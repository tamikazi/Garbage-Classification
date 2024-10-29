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
from torchvision.models import resnet50, ResNet50_Weights

from transformers import BertTokenizer, BertModel
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# Initialize device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# static global variables

# hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
DROPOUT_RATE = 0.2

# dataset directories
TRAINSET_DIR = '/work/TALC/enel645_2024f/garbage_data/CVPR_2024_dataset_Train'
VALSET_DIR = '/work/TALC/enel645_2024f/garbage_data/CVPR_2024_dataset_Val'
TESTSET_DIR = '/work/TALC/enel645_2024f/garbage_data/CVPR_2024_dataset_Test'

# global class to index mapping variables
class_names = ['Green', 'Blue', 'Black', 'TTR']
class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}

# ---------------------------
# Helper Functions and Classes
# ---------------------------

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
        self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        # Remove the last classification layer
        self.model.fc = nn.Identity()
        # Feature extractor to output a feature vector
        self.feature_extractor = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
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

def compute_class_weights(trainset_df):
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

    return class_weights_tensor

def data_preprocessing():
    # Extract the data
    trainset_df = extract_data_from_folders(TRAINSET_DIR)
    valset_df = extract_data_from_folders(VALSET_DIR)
    testset_df = extract_data_from_folders(TESTSET_DIR)

    class_weights_tensor = compute_class_weights(trainset_df)

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
        image_transform=transform_train,
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
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )
    valloader = DataLoader(
        valset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )
    testloader = DataLoader(
        testset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

    return trainloader, valloader, testloader, class_weights_tensor
# ---------------------------
# Training, Validation, and Testing Functions
# ---------------------------

def train_one_epoch(model, trainloader, criterion, optimizer):
    """Train the model for one epoch."""
    model.train()
    running_loss_combined = 0.0
    running_corrects_combined = 0

    # Define a learning rate scheduler 
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

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

    # Step the scheduler at the end of each epoch
    scheduler.step()

    # Compute epoch loss and accuracy
    epoch_loss_combined = running_loss_combined / len(trainloader.dataset)
    epoch_acc_combined = running_corrects_combined.double() / len(trainloader.dataset)

    return epoch_loss_combined, epoch_acc_combined

def train_validate_model(model, trainloader, valloader, criterion, trainable_params, best_val_loss, num_epochs): 
    # Initialize the optimizer
    optimizer = optim.Adam(trainable_params, lr=LEARNING_RATE, weight_decay=1e-4)

    # Early stopping parameters
    early_stopping_patience = 4
    epochs_no_improve = 0

    # Lists to store metrics for plotting
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    for epoch in range(num_epochs):  # Fixed epochs to 10
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # Training phase
        epoch_loss_combined, epoch_acc_combined = train_one_epoch(
            model, trainloader, criterion, optimizer
        )

        train_losses.append(epoch_loss_combined)
        train_accuracies.append(epoch_acc_combined)

        print(f'Training Combined Loss: {epoch_loss_combined:.4f} '
                f'Acc: {epoch_acc_combined:.4f}')

        # Validation phase
        val_loss_combined, val_acc_combined, precision, recall, f1 = validate(
            model, valloader, criterion
        )

        val_losses.append(val_loss_combined)
        val_accuracies.append(val_acc_combined)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

        print(f'Validation Combined Loss: {val_loss_combined:.4f} '
                f'Acc: {val_acc_combined:.4f}')
        print(f'Validation Precision: {precision:.4f}, '
                f'Recall: {recall:.4f}, F1-score: {f1:.4f}')

        # Early stopping
        if val_loss_combined < best_val_loss:
            best_val_loss = val_loss_combined
            epochs_no_improve = 0
            # Save the best model based on validation loss
            print(f"New best model found! Saving model with validation "
                    f"loss: {best_val_loss:.4f}")
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

    # Plotting results
    filename = "/home/shaakira.gadiwan/assignment2/metrics_plot1.png"
    plot_training_results(train_losses, val_losses, train_accuracies, val_accuracies, precisions, recalls, f1_scores, filename)

    return best_val_loss

def validate(model, valloader, criterion):
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

    # Compute metrics
    precision, recall, f1 = compute_metrics(all_labels, all_preds_combined)

    return val_loss_combined, val_acc_combined, precision, recall, f1

def test(model, testloader):
    """Test the model and log misclassified examples."""
    model.eval()
    test_running_corrects_combined = 0

    all_labels = []
    all_preds_combined = []

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

    # Compute test accuracy
    test_acc_combined = test_running_corrects_combined.double() / len(testloader.dataset)

    # Compute metrics
    precision, recall, f1 = compute_metrics(all_labels, all_preds_combined)

    # Confusion matrix
    conf_mat = confusion_matrix(all_labels, all_preds_combined)

    return test_acc_combined, precision, recall, f1, conf_mat

def plot_conf_mat(conf_mat):
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
    plt.savefig('/home/shaakira.gadiwan/assignment2/confusion_matrix.png') 
    plt.close(fig)


def compute_metrics(all_labels, all_preds_combined):
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

    return precision, recall, f1

def plot_training_results(train_losses, val_losses, train_accuracies, val_accuracies, precisions, recalls, f1_scores, filename):
    # Convert list of tensors to NumPy arrays if they are on GPU
    train_losses = [loss.detach().cpu().numpy() if isinstance(loss, torch.Tensor) else loss for loss in train_losses]
    val_losses = [loss.detach().cpu().numpy() if isinstance(loss, torch.Tensor) else loss for loss in val_losses]
    train_accuracies = [acc.detach().cpu().numpy() if isinstance(acc, torch.Tensor) else acc for acc in train_accuracies]
    val_accuracies = [acc.detach().cpu().numpy() if isinstance(acc, torch.Tensor) else acc for acc in val_accuracies]
    precisions = [precision.detach().cpu().numpy() if isinstance(precision, torch.Tensor) else precision for precision in precisions]
    recalls = [recall.detach().cpu().numpy() if isinstance(recall, torch.Tensor) else recall for recall in recalls]
    f1_scores = [f1.detach().cpu().numpy() if isinstance(f1, torch.Tensor) else f1 for f1 in f1_scores]


    epochs = range(1, len(train_losses) + 1)

    # Check epochs and convert to list if it's a range object
    epochs = list(epochs)  

    # Plot Loss
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.scatter(epochs, train_losses, label='Training Loss')
    plt.scatter(epochs, val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(2, 2, 2)
    plt.scatter(epochs, train_accuracies, label='Training Accuracy')
    plt.scatter(epochs, val_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot Precision
    plt.subplot(2, 2, 3)
    plt.scatter(epochs, precisions, label='Precision')
    plt.title('Validation Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.legend()

    # Plot Recall and F1 Score
    plt.subplot(2, 2, 4)
    plt.scatter(epochs, recalls, label='Recall')
    plt.scatter(epochs, f1_scores, label='F1 Score')
    plt.title('Validation Recall and F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# ---------------------------
# Main Function
# ---------------------------

def main():
    trainloader, valloader, testloader, class_weights_tensor = data_preprocessing()

    # Define the loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # Initialize the model
    model = GarbageClassifier(num_classes=len(class_names), dropout_rate=DROPOUT_RATE).to(device)

    # Freeze and unfreeze layers as needed
    # Image model
    for param in model.image_model.model.parameters():
        param.requires_grad = False
    # for param in model.image_model.model.layer3.parameters():
    #     param.requires_grad = True
    for param in model.image_model.model.layer4.parameters():
        param.requires_grad = True
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
        {'params': model.image_model.model.layer4.parameters(), 'lr': LEARNING_RATE * 0.1},
        {'params': model.image_model.feature_extractor.parameters(), 'lr': LEARNING_RATE},
        # Text model
        {'params': model.text_model.bert.parameters(), 'lr': LEARNING_RATE * 0.1},
        {'params': model.text_model.feature_extractor.parameters(), 'lr': LEARNING_RATE},
        # Fusion and classifier layers
        {'params': model.fusion.parameters(), 'lr': LEARNING_RATE},
    ]

    best_val_loss = float('inf')

    best_val_loss = train_validate_model(model, trainloader, valloader, criterion, trainable_params, best_val_loss, 20)

    # Load the best model and evaluate on the test set
    model.load_state_dict(torch.load('best_model.pth', weights_only=True))

    # Freeze and unfreeze layers as needed
    # Image model
    for param in model.image_model.model.parameters():
        param.requires_grad = False
    for param in model.image_model.model.layer2.parameters():
        param.requires_grad = True
    for param in model.image_model.model.layer3.parameters():
        param.requires_grad = True
    for param in model.image_model.model.layer4.parameters():
        param.requires_grad = True
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
        {'params': model.image_model.model.layer2.parameters(), 'lr': LEARNING_RATE * 0.01},
        {'params': model.image_model.model.layer3.parameters(), 'lr': LEARNING_RATE * 0.01},
        {'params': model.image_model.model.layer4.parameters(), 'lr': LEARNING_RATE * 0.01},
        {'params': model.image_model.feature_extractor.parameters(), 'lr': LEARNING_RATE * 0.1},
        # Text model
        {'params': model.text_model.bert.parameters(), 'lr': LEARNING_RATE * 0.01},
        {'params': model.text_model.feature_extractor.parameters(), 'lr': LEARNING_RATE * 0.1},
        # Fusion and classifier layers
        {'params': model.fusion.parameters(), 'lr': LEARNING_RATE},
    ]

    best_val_loss = train_validate_model(model, trainloader, valloader, criterion, trainable_params, best_val_loss, 5)

    # Load the best model and evaluate on the test set
    model.load_state_dict(torch.load('best_model.pth'))

    test_acc_combined, precision, recall, f1, conf_mat = test(model, testloader)

    print(f'Test Combined Accuracy: {test_acc_combined:.4f}')
    print(f'Test Precision: {precision:.4f}, '
            f'Recall: {recall:.4f}, F1-score: {f1:.4f}')

    plot_conf_mat(conf_mat)
    
if __name__ == '__main__':
    main()