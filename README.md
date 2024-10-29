# MultiModal Garbage Classifier - Pytorch Implementation

## Main Classifier Program
The main program, found in /src/multimodal-garbage-classifier.ipynb, sets up a training and testing pipeline for a text- and image-based multimodal model that classifies garbage images into four categories ('Green', 'Blue', 'Black', 'TTR') based on the City of Calgary guidelines. The following sections describe the breakdown of the key components.

---

### **Custom Dataset Class**

A custom dataset class handles multimodal data, enabling easy loading and preprocessing for both images and text simultaneously.

   - **Initialization**: Accepts a dataframe containing image paths, text descriptions, and labels for a structured dataset.

   - **Text Tokenization**: Tokenizes text using a BERT tokenizer, producing token IDs and attention masks, which are either truncated or padded to a fixed length of 32 tokens for uniformity.

   - **Image Processing**: Images are resized, normalized, and converted to tensors compatible with ResNet-50.

   - **Caching**: Tokenized text is cached to avoid repeated computation for duplicate descriptions.

   - **Output**: The `__getitem__` method returns a dictionary with processed image tensors, tokenized text (input IDs, attention masks), and labels for each sample.

---

### **Model Definitions**

The multimodal model combines a ResNet-50 model for images and a BERT-based transformer for text, both pretrained. 

   - **Image Feature Extractor**: ResNet-50, with frozen layers, extracts 2048-dimensional features from images. These are then reduced to a 512-dimensional vector using a fully-connected layer for compatibility in fusion.

   - **Text Feature Extractor**: The BERT model processes text, extracting the final hidden state of the [CLS] token and passing it through a fully-connected layer to produce a 512-dimensional vector, aligning with the image feature vector.

   - **Fusion and Classification Layers**: The 512-dimensional image and text vectors are concatenated to form a 1024-dimensional vector, which is refined to a 256-dimensional feature through two FC layers with ReLU, dropout, and batch normalization. This final feature is classified with a softmax output layer.

---

### **Metric Plotting and Analysis**

   - **Confusion Matrix Plotting** (`plot_conf_mat`): Visualizes classification performance on the test set by plotting a confusion matrix. Displays actual versus predicted labels as a heatmap and saves the output as `confusion_matrix.png`.

   - **Metric Computation** (`compute_metrics`): Calculates precision, recall, and F1-score across all predictions in a weighted average, handling any zero-division cases.

   - **Training and Validation Results Plotting** (`plot_training_results`): This function visualizes the evolution of key metrics over training epochs:
      - **Training and Validation Loss**: Shows learning progress, comparing loss between training and validation sets.
      - **Training and Validation Accuracy**: Highlights accuracy improvements, comparing across training and validation sets.
      - **Precision, Recall, and F1-Score**: Precision is plotted independently; recall and F1-score are combined, all reflecting model effectiveness on validation data. 

---
## Weights and Biases Tooling Program
A useful tool, found in /tool/wandb-garbage-classifier.py, that leverages Weights & Biases (WandB) for hyperparameter tuning through sweeps, aimed at identifying the optimal hyperparameters to enhance model performance.

## Sweep Configuration

### Initialization
The sweep is initialized using `wandb.init()` in the `train()` function, setting up a new run for tracking metrics and configurations.

### Hyperparameter Configuration
Define hyperparameters to sweep such as:
   - `batch_size`: Number of samples per update.
   - `learning_rate`: Step size for optimizing the loss function.
   - `dropout_rate`: Probability of dropping neurons to prevent overfitting.

A desired min and max sweep value can be set for each hyperparameter. 

