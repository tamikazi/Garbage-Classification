# pip install -r src/requirements.txt

import os
import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim

import torchvision
from torchvision import transforms, datasets, models

from torchvision.models import resnet34

from transformers import BertTokenizer
from transformers import BertModel

from sklearn.utils.class_weight import compute_class_weight

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# define static global variables
class_names = ['Green', 'Blue', 'Black', 'TTR']
num_classes = len(class_names)

'''trainset_dir = 'data/enel645_2024f/garbage_data/CVPR_2024_dataset_Train'
valset_dir = 'data/enel645_2024f/garbage_data/CVPR_2024_dataset_Val'
testset_dir = 'data/enel645_2024f/garbage_data/CVPR_2024_dataset_Test'
'''

'''trainset_dir = 'C:/Users/Shaakira Gadiwan/Documents/enel645/Garbage-Classification/data/garbage_data/CVPR_2024_dataset_Train'
valset_dir = 'C:/Users/Shaakira Gadiwan/Documents/enel645/Garbage-Classification/data/garbage_data/CVPR_2024_dataset_Val'
testset_dir = 'C:/Users/Shaakira Gadiwan/Documents/enel645/Garbage-Classification/data/garbage_data/CVPR_2024_dataset_Test'
'''

trainset_dir = r"/work/TALC/enel645_2024f/garbage_data/CVPR_2024_dataset_Train"
valset_dir = r"/work/TALC/enel645_2024f/garbage_data/CVPR_2024_dataset_Val"
testset_dir = r"/work/TALC/enel645_2024f/garbage_data/CVPR_2024_dataset_Test"

# define learning rates
initial_learning_rate = 0.001
fine_tuning_learning_rate = 0.0001  # smaller learning rate for fine-tuning

# global class to index mapping variables
class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
idx_to_class = {idx: class_name for idx, class_name in enumerate(class_names)}

from torch.utils.data import Dataset
from torchvision import transforms
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image

# custom dataset for garbage classification, combining image and text data
class GarbageDataset(Dataset):
    def __init__(self, dataframe, image_transform=None, max_len=32, class_to_idx=None):
        """
        Initialize the GarbageDataset.

        Args:
            dataframe (pd.DataFrame): DataFrame containing image paths, text descriptions, and labels.
            image_transform (callable, optional): Transform to be applied to the images.
            max_len (int): Maximum length for text tokenization.
            class_to_idx (dict, optional): Mapping from class names to numeric indices.
        """
        self.dataframe = dataframe
        self.image_transform = image_transform
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_len = max_len
        self.class_to_idx = class_to_idx  # Pass the class mapping

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.dataframe)

    def __getitem__(self, idx):
        """
        Retrieve an item from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the image, tokenized text, and numeric label.
        """
        # Get image path, text description, and label from the dataframe
        img_path = self.dataframe.iloc[idx]['image_path']
        text_desc = self.dataframe.iloc[idx]['text_description']
        label = self.dataframe.iloc[idx]['label']  

        # Load and preprocess the image
        image = Image.open(img_path).convert("RGB")
        if self.image_transform:
            image = self.image_transform(image)

        # Tokenize the text description
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
            'label': torch.tensor(numeric_label, dtype=torch.long)  
        }

# custom activation function (Modified Swish)
class CustomActivation(nn.Module):
    def forward(self, x):
        """
        Apply the custom activation function.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the activation.
        """
        return x * torch.sigmoid(torch.log(1 + torch.abs(x)))

# define the modified ResNet-34 model for feature extraction
class ModifiedResNet34(nn.Module):
    def __init__(self):
        """
        Initialize the Modified ResNet-34 model.

        This model removes the final classification layer and adds a custom activation function.
        """
        super(ModifiedResNet34, self).__init__()
        self.base_model = models.resnet34(pretrained=True)
        self.base_model.fc = nn.Identity()  # Remove the final classification layer
        self.activation = CustomActivation()

    def forward(self, x):
        """
        Forward pass through the modified ResNet-34.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Extracted features from the input image.
        """
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.activation(x)
        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)
        x = self.base_model.avgpool(x)
        x = torch.flatten(x, 1)  # Resulting in a [batch_size, 512] tensor
        return x

# garbage classifier that combines image and text features
class GarbageClassifier(nn.Module):
    def __init__(self, num_classes=4):
        """
        Initialize the GarbageClassifier.

        Args:
            num_classes (int): Number of output classes for classification.
        """
        super(GarbageClassifier, self).__init__()
        # image feature extraction with ResNet
        self.resnet = ModifiedResNet34()

        # text feature extraction with BERT
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # classification layer after combining image and text features
        self.fc_combined = nn.Linear(768 + 512, num_classes)

        # separate classification layers for image and text outputs
        self.fc_image = nn.Linear(512, num_classes)
        self.fc_text = nn.Linear(768, num_classes)

    def forward(self, image, input_ids, attention_mask):
        """
        Forward pass through the GarbageClassifier.

        Args:
            image (torch.Tensor): Input image tensor.
            input_ids (torch.Tensor): Input token IDs for text.
            attention_mask (torch.Tensor): Attention mask for the text input.

        Returns:
            tuple: Separate and combined outputs for image and text classification.
        """
        # get image features from ResNet
        image_features = self.resnet(image)

        # get text features from BERT
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = bert_output.pooler_output  # (batch_size, 768)

        # separate predictions for image and text
        image_output = self.fc_image(image_features)  # classification based on image alone
        text_output = self.fc_text(text_features)  # classification based on text alone

        # combined features from both image and text
        combined_features = torch.cat((image_features, text_features), dim=1)
        combined_output = self.fc_combined(combined_features)

        return image_output, text_output, combined_output
    
def extract_data_from_folders(base_dir):
    """
    Extract images, labels, and text descriptions from a given folder.

    Args:
    - base_dir: The base directory containing subfolders with images.

    Returns:
    - A pandas DataFrame with columns 'image_path', 'text_description', and 'label'.
    """
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
                        'label': label_folder  # The subfolder name represents the label (bin)
                    })

    # Convert to DataFrame for easy manipulation
    return pd.DataFrame(data)

def get_dataset_stats(dataloader):
    """
    Calculate dataset statistics (mean and std) for normalization.

    Args:
    - dataloader: A DataLoader for the dataset.

    Returns:
    - mean: The mean of the dataset across channels.
    - std: The standard deviation of the dataset across channels.
    """
    mean = 0.
    std = 0.
    nb_samples = 0.

    for batch in dataloader:
        # get the images from the batch
        images = batch['image']  # accessing the image tensor
        batch_samples = images.size(0)  # number of samples in the batch
        images = images.view(batch_samples, images.size(1), -1)  # reshape to (batch_size, channels, height * width)
        
        mean += images.mean(2).sum(0)  # accumulate mean for each channel
        std += images.std(2).sum(0)    # accumulate std for each channel
        nb_samples += batch_samples      # total number of samples

    mean /= nb_samples  # calculate overall mean
    std /= nb_samples    # calculate overall std
    return mean, std

def display_sample_from_trainset(trainset, idx=0):
    """
    Display an image sample from the training set, along with its label and text description.

    Args:
    - trainset: The dataset containing the samples.
    - idx: Index of the sample to display (default: 0).
    """
    # Extract the image, label, and input IDs directly from the dataset
    sample = trainset[idx]
    image = sample['image']
    label_idx = sample['label'].item()  # Convert to int if needed
    input_ids = sample['input_ids']

    # Convert label index to label string (ensure idx_to_class is defined)
    label = idx_to_class[label_idx]

    # Decode the text description from input IDs (ensure tokenizer is available)
    text_description = trainset.tokenizer.decode(input_ids, skip_special_tokens=True)

    # Unnormalize the image (use the same mean and std from your transforms)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image * std + mean  # Undo normalization

    # Permute the image to match the (H, W, C) format for matplotlib
    image = image.permute(1, 2, 0).numpy()

    # Display the image along with label and text description
    plt.imshow(image)
    plt.title(f'Class: {label}\nText: {text_description}')
    plt.axis('off')
    plt.show()  

def train_one_epoch(model, trainloader, optimizer, criterion, device):
    """
    Train the model for one epoch, calculating losses and accuracies for image, text, and combined outputs.

    Args:
    - model: The model to train.
    - trainloader: DataLoader for the training set.
    - optimizer: The optimizer for updating model parameters.
    - criterion: The loss function.
    - device: The device on which to perform training (CPU or GPU).

    Returns:
    - Tuple of losses and accuracies for image, text, and combined outputs.
    """
    running_loss_image = 0.0
    running_loss_text = 0.0
    running_loss_combined = 0.0
    running_corrects_image = 0
    running_corrects_text = 0
    running_corrects_combined = 0

    model.train()
    for batch in trainloader:
        images = batch['image'].to(device)
        text_input_ids = batch['input_ids'].to(device)
        text_attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()

        # Forward pass (separate outputs)
        image_output, text_output, combined_output = model(images, text_input_ids, text_attention_mask)
        
        # Compute loss for image, text, and combined models
        loss_image = criterion(image_output, labels)
        loss_text = criterion(text_output, labels)
        loss_combined = criterion(combined_output, labels)
        
        # Backward pass and optimization
        total_loss = loss_image + loss_text + loss_combined
        total_loss.backward()
        optimizer.step()

        # Update running loss and accuracy for image, text, and combined
        running_loss_image += loss_image.item() * images.size(0)
        running_loss_text += loss_text.item() * images.size(0)
        running_loss_combined += loss_combined.item() * images.size(0)
        
        _, preds_image = torch.max(image_output, 1)
        _, preds_text = torch.max(text_output, 1)
        _, preds_combined = torch.max(combined_output, 1)

        running_corrects_image += torch.sum(preds_image == labels.data)
        running_corrects_text += torch.sum(preds_text == labels.data)
        running_corrects_combined += torch.sum(preds_combined == labels.data)
    
    # Compute epoch loss and accuracy for image, text, and combined
    epoch_loss_image = running_loss_image / len(trainset)
    epoch_loss_text = running_loss_text / len(trainset)
    epoch_loss_combined = running_loss_combined / len(trainset)
    
    epoch_acc_image = running_corrects_image.double() / len(trainset)
    epoch_acc_text = running_corrects_text.double() / len(trainset)
    epoch_acc_combined = running_corrects_combined.double() / len(trainset)

    return (epoch_loss_image, epoch_acc_image, epoch_loss_text, 
            epoch_acc_text, epoch_loss_combined, epoch_acc_combined)

def validate_model(model, valloader, criterion, device):
    """
    Validate the model on the validation set, calculating losses and accuracies.

    Args:
    - model: The model to validate.
    - valloader: DataLoader for the validation set.
    - criterion: The loss function.
    - device: The device on which to perform validation (CPU or GPU).

    Returns:
    - Tuple of losses and accuracies for image, text, and combined outputs.
    """
    val_running_loss_image = 0.0
    val_running_loss_text = 0.0
    val_running_loss_combined = 0.0
    val_running_corrects_image = 0
    val_running_corrects_text = 0
    val_running_corrects_combined = 0
    
    model.eval()
    with torch.no_grad():
        for batch in valloader:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # Forward pass (separate outputs)
            image_output, text_output, combined_output = model(images, input_ids, attention_mask)
            
            # Compute loss for image, text, and combined models
            loss_image = criterion(image_output, labels)
            loss_text = criterion(text_output, labels)
            loss_combined = criterion(combined_output, labels)
            
            # Update running loss and accuracy for image, text, and combined
            val_running_loss_image += loss_image.item() * images.size(0)
            val_running_loss_text += loss_text.item() * images.size(0)
            val_running_loss_combined += loss_combined.item() * images.size(0)
            
            _, preds_image = torch.max(image_output, 1)
            _, preds_text = torch.max(text_output, 1)
            _, preds_combined = torch.max(combined_output, 1)

            val_running_corrects_image += torch.sum(preds_image == labels.data)
            val_running_corrects_text += torch.sum(preds_text == labels.data)
            val_running_corrects_combined += torch.sum(preds_combined == labels.data)

    # Compute validation loss and accuracy for image, text, and combined
    val_loss_image = val_running_loss_image / len(valset)
    val_loss_text = val_running_loss_text / len(valset)
    val_loss_combined = val_running_loss_combined / len(valset)
    
    val_acc_image = val_running_corrects_image.double() / len(valset)
    val_acc_text = val_running_corrects_text.double() / len(valset)
    val_acc_combined = val_running_corrects_combined.double() / len(valset)

    return (val_loss_image, val_acc_image, val_loss_text, 
            val_acc_text, val_loss_combined, val_acc_combined)

def test_model(model, testloader, testset, device):
    """
    Function to test the model and calculate accuracy on the test set.

    Args:
    - model: Trained PyTorch model
    - testloader: DataLoader for the test dataset
    - testset: The test dataset
    - device: The device on which to perform testing (CPU or GPU)

    Returns:
    - test_acc: Accuracy of the model on the test set
    """
    model.eval()  # Set the model to evaluation mode

    test_running_corrects = 0

    with torch.no_grad():  # Disable gradient computation for testing
        for batch in testloader:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # Get outputs from the model
            image_output, text_output, combined_output = model(images, input_ids=input_ids, attention_mask=attention_mask)

            # Use combined_output for prediction
            _, preds = torch.max(combined_output, 1)

            # Update running corrects
            test_running_corrects += torch.sum(preds == labels.data)

    # Calculate test accuracy
    test_acc = test_running_corrects.double() / len(testset)
    print(f'Test Accuracy: {test_acc:.4f}')

    return test_acc

if __name__ == '__main__':
    # init
    batch_size = 32
    num_workers = 4

    run = wandb.init(project='garbage-collection')

    # initialize the transforms
    torchvision_transform = transforms.Compose([transforms.Resize((224,224)),\
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
        transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406] ,std=[0.229, 0.224, 0.225] )])

    torchvision_transform_test = transforms.Compose([transforms.Resize((224,224)),\
        transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406] ,std=[0.229, 0.224, 0.225])])

    # extract the data
    trainset_df = extract_data_from_folders(trainset_dir)
    valset_df = extract_data_from_folders(valset_dir)
    testset_df = extract_data_from_folders(testset_dir)

    # transform and normalize the training, validation, and testing data
    trainset = GarbageDataset(trainset_df, image_transform=torchvision_transform, class_to_idx=class_to_idx)
    valset = GarbageDataset(valset_df, image_transform=torchvision_transform, class_to_idx=class_to_idx)
    testset = GarbageDataset(testset_df, image_transform=torchvision_transform_test, class_to_idx=class_to_idx)

    # create the dataloaders
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    #display_sample_from_trainset(trainset, idx=0)

    # fix the class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(trainset_df['label']), y=trainset_df['label'])
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # initialize the model and freeze the desired layers
    model = GarbageClassifier(num_classes=len(class_names)).to(device)

    # freeze all ResNet layers and unfreeze the last ResNet block
    for name, param in model.resnet.named_parameters():
        if 'layer4' in name:
            param.requires_grad = True  # Unfreeze last ResNet block
        else:
            param.requires_grad = False  # Freeze other layers

    # freeze all BERT layers except the last BERT encoder layer
    for i, param in enumerate(model.bert.parameters()):
        param.requires_grad = False  # Freeze all parameters
        if i >= len(model.bert.encoder.layer) - 1:  # Check if it's the last encoder layer
            param.requires_grad = True  # Unfreeze parameters of the last encoder layer

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=initial_learning_rate)

    num_epochs = 25
    wandb.config = {"epochs": num_epochs, "batch_size": batch_size, "learning_rate": initial_learning_rate}

    best_val_acc = 0.0  # Variable to track the best validation accuracy

    # Main training loop
    for epoch in range(wandb.config['epochs']):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # Train for one epoch
        train_results = train_one_epoch(model, trainloader, optimizer, criterion, device)
        epoch_loss_image, epoch_acc_image, epoch_loss_text, epoch_acc_text, epoch_loss_combined, epoch_acc_combined = train_results

        # Validate the model
        val_results = validate_model(model, valloader, criterion, device)
        val_loss_image, val_acc_image, val_loss_text, val_acc_text, val_loss_combined, val_acc_combined = val_results

        # Log results
        wandb.log({
            "Training Loss (Image)": epoch_loss_image, 
            "Training Accuracy (Image)": epoch_acc_image,
            "Training Loss (Text)": epoch_loss_text, 
            "Training Accuracy (Text)": epoch_acc_text,
            "Training Loss (Combined)": epoch_loss_combined, 
            "Training Accuracy (Combined)": epoch_acc_combined,
            "Validation Loss (Image)": val_loss_image, 
            "Validation Accuracy (Image)": val_acc_image,
            "Validation Loss (Text)": val_loss_text, 
            "Validation Accuracy (Text)": val_acc_text,
            "Validation Loss (Combined)": val_loss_combined, 
            "Validation Accuracy (Combined)": val_acc_combined
        })

        if val_acc_combined > best_val_acc:
            best_val_acc = val_acc_combined
            print(f"New best model found! Saving model with validation accuracy: {best_val_acc:.4f}")
            torch.save(model.state_dict(), './best_garbage_model.pth') 

    model.load_state_dict(torch.load('./best_garbage_model.pth'))  # Load the best model from the initial training

    # Unfreeze the desired layers for fine-tuning
    for param in model.resnet.parameters():
        param.requires_grad = True  # Unfreeze all ResNet layers or selectively as per your choice

    for param in model.bert.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=fine_tuning_learning_rate)
    wandb.config.update({"fine_tuning_epochs": num_epochs, "fine_tuning_learning_rate": fine_tuning_learning_rate})

    best_val_acc_fine_tuning = 0.0

    # Fine training loop
    for epoch in range(wandb.config['epochs']):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # Train for one epoch
        train_results = train_one_epoch(model, trainloader, optimizer, criterion, device)
        epoch_loss_image, epoch_acc_image, epoch_loss_text, epoch_acc_text, epoch_loss_combined, epoch_acc_combined = train_results

        # Validate the model
        val_results = validate_model(model, valloader, criterion, device)
        val_loss_image, val_acc_image, val_loss_text, val_acc_text, val_loss_combined, val_acc_combined = val_results

        # Log results
        wandb.log({
            "Training Loss (Image)": epoch_loss_image, 
            "Training Accuracy (Image)": epoch_acc_image,
            "Training Loss (Text)": epoch_loss_text, 
            "Training Accuracy (Text)": epoch_acc_text,
            "Training Loss (Combined)": epoch_loss_combined, 
            "Training Accuracy (Combined)": epoch_acc_combined,
            "Validation Loss (Image)": val_loss_image, 
            "Validation Accuracy (Image)": val_acc_image,
            "Validation Loss (Text)": val_loss_text, 
            "Validation Accuracy (Text)": val_acc_text,
            "Validation Loss (Combined)": val_loss_combined, 
            "Validation Accuracy (Combined)": val_acc_combined
        })

        if val_acc_combined > best_val_acc_fine_tuning:
            best_val_acc_fine_tuning = val_acc_combined
            print(f"New best model found! Saving model with validation accuracy: {best_val_acc:.4f}")
            torch.save(model.state_dict(), './best_garbage_model.pth') 
        
    model.load_state_dict(torch.load('./best_garbage_model.pth'))
    test_accuracy = test_model(model, testloader, testset, device)
    wandb.log({"Test Accuracy": test_accuracy})