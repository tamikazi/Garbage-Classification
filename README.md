# MultiModal Garbage Classifier - Pytorch Implementation

## Main Classifier Program
The main program, found in /src/multimodal-garbage-classifier.ipynb, sets up a training and testing pipeline for a text- and image-based multimodal model that classifies garbage images into four categories ('Green', 'Blue', 'Black', 'TTR') based on the [City of Calgary guidelines](https://www.calgary.ca/waste/what-goes-where/default.html). 

### How to run the main program:
- ...

---

## Summary of Key Features

### Custom Dataset Class
A dataset class loads and preprocesses both images and text. It:
   - Initializes with a structured dataset of image paths, text, and labels.
   - Tokenizes text to 32 tokens using BERT, with padding and truncation.
   - Preprocesses images for ResNet-50.
   - Caches tokenized text to avoid redundant processing.
   - Outputs a dictionary of image tensors, tokenized text, and labels.

---

### Model Design

The multimodal model combines ResNet-50 for images and BERT for text:
   - **Image**: ResNet-50 extracts 2048-dimensional features, reduced to 512 for fusion.
   - **Text**: BERT processes text, outputting a 512-dimensional vector from the [CLS] token.
   - **Fusion and Classification**: Combines image and text features into a 1024-dimensional vector, then refines to 256 dimensions before classifying with softmax.
  
---

### **Metrics and Plotting**
Functions track and visualize model performance:
   - **Confusion Matrix**: Heatmap showing test predictions.
   - **Metrics Calculation**: Computes precision, recall, and F1-score.
   - **Training Results Plotting**: Shows loss, accuracy, precision, recall, and F1-score trends across epochs.
   
---

## Weights and Biases Tooling Program
A useful tool, found in /tool/wandb-garbage-classifier.py, that leverages Weights & Biases (WandB) for hyperparameter tuning through sweeps, aimed at identifying the optimal hyperparameters to enhance model performance.

### How to run the tooling program:
- ...

---

## Sweep Configuration

### Initialization
The sweep is initialized using `wandb.init()` in the `train()` function, setting up a new run for tracking metrics and configurations.

### Hyperparameter Configuration
Define hyperparameters to sweep such as:
   - `batch_size`: Number of samples per update.
   - `learning_rate`: Step size for optimizing the loss function.
   - `dropout_rate`: Probability of dropping neurons to prevent overfitting.

A desired min and max sweep value can be set for each hyperparameter. 