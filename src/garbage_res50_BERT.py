import os
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

from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image

from transformers import BertTokenizer, BertModel
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

import wandb

# Initialize wandb with project settings
wandb.init(
    project='garbage-collection',
    config={
        "epochs": 10,
        "batch_size": 512,
        "learning_rate": 0.001,
        "max_len": 32,
        "optimizer": "Adam",
        "model_architecture": "ResNet50+BERT",
    },
    settings=wandb.Settings(code_dir='.')
)

config = wandb.config

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


trainset_dir = '/work/TALC/enel645_2024f/garbage_data/CVPR_2024_dataset_Train'
valset_dir = '/work/TALC/enel645_2024f/garbage_data/CVPR_2024_dataset_Val'
testset_dir = '/work/TALC/enel645_2024f/garbage_data/CVPR_2024_dataset_Test'

# Function to extract data from folders
def extract_data_from_folders(base_dir):
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
                    data.append({
                        'image_path': image_path,
                        'text_description': text_description,
                        'label': label_folder  # The subfolder name represents the label
                    })
    # Convert to DataFrame for easy manipulation
    return pd.DataFrame(data)

class GarbageDataset(Dataset):
    def __init__(self, dataframe, image_transform=None, max_len=32,
                 tokenizer=None, class_to_idx=None):
        self.dataframe = dataframe
        self.image_transform = image_transform
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.class_to_idx = class_to_idx


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

        # Tokenize the text description using BERT tokenizer
        text_inputs = self.tokenizer(
            text_desc,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

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
    def __init__(self):
        super(ImageModel, self).__init__()
        self.model = models.resnet50(pretrained=True)
        # Remove the last classification layer
        self.model.fc = nn.Identity()
        # Feature extractor to output a feature vector of size 512
        self.feature_extractor = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        # Placeholder for features
        self.features_output = None

        # Register a hook to save features from layer4
        self.model.layer4.register_forward_hook(self.save_features)

    def save_features(self, module, input, output):
        self.features_output = output  # Save the output of layer4

    def forward(self, x):
        x = self.model(x)  # x will have shape (batch_size, 2048)
        x = self.feature_extractor(x)  # x will have shape (batch_size, 512)
        return x

# Define the text model using BERT
class TextModel(nn.Module):
    def __init__(self, pretrained_model_name='bert-base-uncased'):
        super(TextModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        # Feature extractor to output a feature vector of size 512
        self.feature_extractor = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, input_ids, attention_mask):
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # Shape: (batch_size, 768)
        x = self.feature_extractor(pooled_output)  # Shape: (batch_size, 512)
        return x  # Output feature vector of size 512

# Modified GarbageClassifier to output individual predictions
class GarbageClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(GarbageClassifier, self).__init__()
        # Image feature extraction with ResNet-50
        self.image_model = ImageModel()
        # Text feature extraction with BERT
        self.text_model = TextModel()
        # Individual classifiers
        self.image_classifier = nn.Linear(512, num_classes)
        self.text_classifier = nn.Linear(512, num_classes)
        # Fusion and classification layers
        self.fusion = nn.Sequential(
            nn.Linear(512 + 512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, input_ids, attention_mask):
        # Get image features
        image_features = self.image_model(image)  # Shape: (batch_size, 512)
        # Get text features
        text_features = self.text_model(input_ids, attention_mask)  # Shape: (batch_size, 512)
        # Individual predictions
        image_output = self.image_classifier(image_features)  # Shape: (batch_size, num_classes)
        text_output = self.text_classifier(text_features)     # Shape: (batch_size, num_classes)
        # Combined features
        combined_features = torch.cat((image_features, text_features), dim=1)  # Shape: (batch_size, 1024)
        # Combined prediction
        combined_output = self.fusion(combined_features)      # Shape: (batch_size, num_classes)
        return combined_output, image_output, text_output

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
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Create datasets
trainset = GarbageDataset(
    trainset_df,
    image_transform=transform,
    class_to_idx=class_to_idx,
    tokenizer=tokenizer,
    max_len=config.max_len
)
valset = GarbageDataset(
    valset_df,
    image_transform=transform,
    class_to_idx=class_to_idx,
    tokenizer=tokenizer,
    max_len=config.max_len
)
testset = GarbageDataset(
    testset_df,
    image_transform=transform,
    class_to_idx=class_to_idx,
    tokenizer=tokenizer,
    max_len=config.max_len
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

# Get the class labels
class_labels = np.unique(trainset_df['label'])

# Compute class weights
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

# Initialize the model, loss function, and optimizer
model = GarbageClassifier(num_classes=len(class_names)).to(device)

# Freeze and unfreeze layers as needed
# Image model
for param in model.image_model.model.parameters():
    param.requires_grad = False

for param in model.image_model.model.layer3.parameters():
    param.requires_grad = True

for param in model.image_model.model.layer4.parameters():
    param.requires_grad = True
for param in model.image_model.feature_extractor.parameters():
    param.requires_grad = True
# Text model
for param in model.text_model.bert.parameters():
    param.requires_grad = False
for param in model.text_model.bert.encoder.layer[-1:].parameters():
    param.requires_grad = True
for param in model.text_model.feature_extractor.parameters():
    param.requires_grad = True


optimizer = optim.Adam([
    {'params': model.image_model.model.layer3.parameters(), 'lr': config.learning_rate * 0.1},
    {'params': model.image_model.model.layer4.parameters(), 'lr': config.learning_rate * 0.1},
    {'params': model.image_model.feature_extractor.parameters(), 'lr': config.learning_rate},
    # Add text model parameters similarly
    {'params': model.text_model.feature_extractor.parameters(), 'lr': config.learning_rate},
    {'params': model.fusion.parameters(), 'lr': config.learning_rate},
    {'params': model.image_classifier.parameters(), 'lr': config.learning_rate},
    {'params': model.text_classifier.parameters(), 'lr': config.learning_rate},
], lr=config.learning_rate)

# Watch the model (logs gradients and parameters)
wandb.watch(model, log="all", log_freq=100)

best_val_acc = 0.0  # Variable to track the best validation accuracy
previous_val_loss = float('inf')  # For wandb alert

for epoch in range(config.epochs):
    print(f'Epoch {epoch + 1}/{config.epochs}')
    print('-' * 10)
    
    # Training phase
    model.train()
    running_loss_combined = 0.0
    running_corrects_combined = 0
    running_loss_image = 0.0
    running_corrects_image = 0
    running_loss_text = 0.0
    running_corrects_text = 0

    for i, batch in enumerate(trainloader):
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()

        # Forward pass
        combined_outputs, image_outputs, text_outputs = model(
            images, input_ids, attention_mask
        )
        
        # Compute losses
        loss_combined = criterion(combined_outputs, labels)
        loss_image = criterion(image_outputs, labels)
        loss_text = criterion(text_outputs, labels)
        
        # Total loss
        total_loss = loss_combined + loss_image + loss_text
        
        # Backward pass and optimization
        total_loss.backward()
        optimizer.step()

        # Update running losses
        running_loss_combined += loss_combined.item() * images.size(0)
        running_loss_image += loss_image.item() * images.size(0)
        running_loss_text += loss_text.item() * images.size(0)

        # Predictions
        _, preds_combined = torch.max(combined_outputs, 1)
        _, preds_image = torch.max(image_outputs, 1)
        _, preds_text = torch.max(text_outputs, 1)

        # Update running corrects
        running_corrects_combined += torch.sum(preds_combined == labels.data)
        running_corrects_image += torch.sum(preds_image == labels.data)
        running_corrects_text += torch.sum(preds_text == labels.data)

    # Compute epoch losses and accuracies
    epoch_loss_combined = running_loss_combined / len(trainset)
    epoch_acc_combined = running_corrects_combined.double() / len(trainset)

    epoch_loss_image = running_loss_image / len(trainset)
    epoch_acc_image = running_corrects_image.double() / len(trainset)

    epoch_loss_text = running_loss_text / len(trainset)
    epoch_acc_text = running_corrects_text.double() / len(trainset)

    print(f'Training Combined Loss: {epoch_loss_combined:.4f} '
          f'Acc: {epoch_acc_combined:.4f}')
    print(f'Training Image Loss: {epoch_loss_image:.4f} '
          f'Acc: {epoch_acc_image:.4f}')
    print(f'Training Text Loss: {epoch_loss_text:.4f} '
          f'Acc: {epoch_acc_text:.4f}')

    wandb.log({
        "Epoch": epoch + 1,
        "Training Combined Loss": epoch_loss_combined,
        "Training Combined Accuracy": epoch_acc_combined.item(),
        "Training Image Loss": epoch_loss_image,
        "Training Image Accuracy": epoch_acc_image.item(),
        "Training Text Loss": epoch_loss_text,
        "Training Text Accuracy": epoch_acc_text.item()
    })

    # Validation phase
    model.eval()
    val_running_loss_combined = 0.0
    val_running_corrects_combined = 0
    val_running_loss_image = 0.0
    val_running_corrects_image = 0
    val_running_loss_text = 0.0
    val_running_corrects_text = 0

    all_labels = []
    all_preds_combined = []
    all_preds_image = []
    all_preds_text = []

    with torch.no_grad():
        for batch in valloader:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # Forward pass
            combined_outputs, image_outputs, text_outputs = model(
                images, input_ids, attention_mask
            )
            
            # Compute losses
            loss_combined = criterion(combined_outputs, labels)
            loss_image = criterion(image_outputs, labels)
            loss_text = criterion(text_outputs, labels)
            
            # Update running losses
            val_running_loss_combined += loss_combined.item() * images.size(0)
            val_running_loss_image += loss_image.item() * images.size(0)
            val_running_loss_text += loss_text.item() * images.size(0)

            # Predictions
            _, preds_combined = torch.max(combined_outputs, 1)
            _, preds_image = torch.max(image_outputs, 1)
            _, preds_text = torch.max(text_outputs, 1)

            # Update running corrects
            val_running_corrects_combined += torch.sum(
                preds_combined == labels.data
            )
            val_running_corrects_image += torch.sum(
                preds_image == labels.data
            )
            val_running_corrects_text += torch.sum(
                preds_text == labels.data
            )

            # Collect all labels and predictions for metrics
            all_labels.extend(labels.cpu().numpy())
            all_preds_combined.extend(preds_combined.cpu().numpy())
            all_preds_image.extend(preds_image.cpu().numpy())
            all_preds_text.extend(preds_text.cpu().numpy())

    # Compute validation losses and accuracies
    val_loss_combined = val_running_loss_combined / len(valset)
    val_acc_combined = val_running_corrects_combined.double() / len(valset)

    val_loss_image = val_running_loss_image / len(valset)
    val_acc_image = val_running_corrects_image.double() / len(valset)

    val_loss_text = val_running_loss_text / len(valset)
    val_acc_text = val_running_corrects_text.double() / len(valset)

    # Compute additional metrics
    precision = precision_score(
        all_labels, all_preds_combined, average='weighted'
    )
    recall = recall_score(
        all_labels, all_preds_combined, average='weighted'
    )
    f1 = f1_score(
        all_labels, all_preds_combined, average='weighted'
    )

    print(f'Validation Combined Loss: {val_loss_combined:.4f} '
          f'Acc: {val_acc_combined:.4f}')
    print(f'Validation Image Loss: {val_loss_image:.4f} '
          f'Acc: {val_acc_image:.4f}')
    print(f'Validation Text Loss: {val_loss_text:.4f} '
          f'Acc: {val_acc_text:.4f}')
    print(f'Validation Precision: {precision:.4f}, '
          f'Recall: {recall:.4f}, F1-score: {f1:.4f}')

    wandb.log({
        "Epoch": epoch + 1,
        "Validation Combined Loss": val_loss_combined,
        "Validation Combined Accuracy": val_acc_combined.item(),
        "Validation Image Loss": val_loss_image,
        "Validation Image Accuracy": val_acc_image.item(),
        "Validation Text Loss": val_loss_text,
        "Validation Text Accuracy": val_acc_text.item(),
        "Validation Precision": precision,
        "Validation Recall": recall,
        "Validation F1-score": f1
    })

    # Confusion matrix
    conf_mat = confusion_matrix(all_labels, all_preds_combined)
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
    plt.title('Validation Confusion Matrix')
    wandb.log({"Validation Confusion Matrix": wandb.Image(fig)})
    plt.close(fig)

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

    # wandb alert for increasing validation loss
    if val_loss_combined > previous_val_loss:
        wandb.alert(
            title="Validation Loss Increase",
            text=f"Validation loss increased from {previous_val_loss:.4f} "
                 f"to {val_loss_combined:.4f} at epoch {epoch+1}",
            level=wandb.AlertLevel.WARN
        )
    previous_val_loss = val_loss_combined

# Load the best model and evaluate on the test set
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
test_running_corrects_combined = 0
test_running_corrects_image = 0
test_running_corrects_text = 0

all_labels = []
all_preds_combined = []
all_preds_image = []
all_preds_text = []

misclassified_table = wandb.Table(
    columns=["Image", "Predicted Label", "True Label", "Text Description"]
)

with torch.no_grad():
    for batch in testloader:
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        text_descriptions = batch['text_description']

        # Forward pass
        combined_outputs, image_outputs, text_outputs = model(
            images, input_ids, attention_mask
        )

        # Predictions
        _, preds_combined = torch.max(combined_outputs, 1)
        _, preds_image = torch.max(image_outputs, 1)
        _, preds_text = torch.max(text_outputs, 1)

        test_running_corrects_combined += torch.sum(
            preds_combined == labels.data
        )
        test_running_corrects_image += torch.sum(
            preds_image == labels.data
        )
        test_running_corrects_text += torch.sum(
            preds_text == labels.data
        )

        # Collect all labels and predictions for metrics
        all_labels.extend(labels.cpu().numpy())
        all_preds_combined.extend(preds_combined.cpu().numpy())
        all_preds_image.extend(preds_image.cpu().numpy())
        all_preds_text.extend(preds_text.cpu().numpy())

        # Log misclassified examples
        for img, pred_label, true_label, text in zip(
            images.cpu(), preds_combined.cpu(), labels.cpu(), text_descriptions
        ):
            if pred_label != true_label:
                img_pil = transforms.ToPILImage()(img)
                misclassified_table.add_data(
                    wandb.Image(img_pil),
                    idx_to_class[pred_label.item()],
                    idx_to_class[true_label.item()],
                    text
                )

# Compute test accuracies
test_acc_combined = test_running_corrects_combined.double() / len(testset)
test_acc_image = test_running_corrects_image.double() / len(testset)
test_acc_text = test_running_corrects_text.double() / len(testset)

# Compute additional metrics
precision = precision_score(
    all_labels, all_preds_combined, average='weighted'
)
recall = recall_score(
    all_labels, all_preds_combined, average='weighted'
)
f1 = f1_score(
    all_labels, all_preds_combined, average='weighted'
)

print(f'Test Combined Accuracy: {test_acc_combined:.4f}')
print(f'Test Image Accuracy: {test_acc_image:.4f}')
print(f'Test Text Accuracy: {test_acc_text:.4f}')
print(f'Test Precision: {precision:.4f}, '
      f'Recall: {recall:.4f}, F1-score: {f1:.4f}')

wandb.log({
    "Test Combined Accuracy": test_acc_combined.item(),
    "Test Image Accuracy": test_acc_image.item(),
    "Test Text Accuracy": test_acc_text.item(),
    "Test Precision": precision,
    "Test Recall": recall,
    "Test F1-score": f1,
    "Misclassified Examples": misclassified_table
})

# Confusion matrix for test set
conf_mat = confusion_matrix(all_labels, all_preds_combined)
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

# Initialize the CAM extractor
cam_extractor = SmoothGradCAMpp(model.image_model.model, target_layer='layer4')

# Generate Grad-CAM visualizations
model.eval()
gradcam_images = []


for batch in testloader:
    images = batch['image'].to(device)
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['label'].to(device)
    text_descriptions = batch['text_description']

    # Forward pass through the combined model
    combined_outputs, image_outputs, text_outputs = model(
        images, input_ids, attention_mask
    )

    # Predictions
    _, preds_combined = torch.max(combined_outputs, 1)

    for idx in range(images.size(0)):
        # Limit to a subset of images
        if len(gradcam_images) >= 10:
            break

        # Get the image and label
        image = images[idx].unsqueeze(0)  # Add batch dimension
        label = labels[idx]
        pred_label = preds_combined[idx]

        # Perform forward pass to get features
        _ = model.image_model(image)

        # Generate CAM
        activation_map = cam_extractor(pred_label.item(), model.image_model.features_output)

        # Convert image to PIL format
        img = images[idx].cpu()
        img_pil = transforms.ToPILImage()(img)

        # Overlay CAM on image
        result = overlay_mask(img_pil, transforms.ToPILImage()(activation_map[0]), alpha=0.5)

        # Log the image with wandb
        gradcam_images.append(
            wandb.Image(result, caption=f"Predicted: {idx_to_class[pred_label.item()]}, Actual: {idx_to_class[label.item()]}")
        )

    if len(gradcam_images) >= 10:
        break  # Only process a subset

# Log Grad-CAM images to wandb
wandb.log({"Grad-CAM Images": gradcam_images})

wandb.finish()
