This code is setting up a training and testing pipeline for a model that classifies garbage images into four categories ('Green', 'Blue', 'Black', 'TTR'). Here’s a breakdown of the key components:

---

### **Model Definitions**

The multimodal model combines a ResNet-50 for images and a BERT-based transformer for text, both pretrained. 

   - **Image Feature Extractor**: ResNet-50, with frozen layers, extracts 2048-dimensional features from images. These are then reduced to a 512-dimensional vector using a fully connected (FC) layer for compatibility in fusion.

   - **Text Feature Extractor**: The BERT model processes text, extracting the final hidden state of the [CLS] token and passing it through an FC layer to produce a 512-dimensional vector, aligning with the image feature vector.

   - **Fusion and Classification Layers**: The 512-dimensional image and text vectors are concatenated to form a 1024-dimensional vector, which is refined to a 256-dimensional feature through two FC layers with ReLU, dropout, and batch normalization. This final feature is classified with a softmax output layer.

---

### **Custom Dataset Class**

This PyTorch dataset class handles multimodal data, enabling seamless loading and preprocessing for both images and text.

   - **Initialization**: Accepts a dataframe containing image paths, text descriptions, and labels for a structured dataset.

   - **Text Tokenization**: Tokenizes text using a BERT tokenizer, producing token IDs and attention masks, which are either truncated or padded to a fixed length of 32 tokens for uniformity.

   - **Image Processing**: Images are resized, normalized, and converted to tensors compatible with ResNet-50.

   - **Caching**: Tokenized text is cached to avoid repeated computation for duplicate descriptions.

   - **Output**: The `__getitem__` method returns a dictionary with processed image tensors, tokenized text (input IDs, attention masks), and labels for each sample.

---

### **Metric Plotting for Training and Evaluation**

Accuracy and loss are tracked for each epoch to monitor model performance.

   - **Accuracy and Loss Tracking**: Training and validation metrics are stored, allowing comparison across epochs.

   - **Visualization**: Using Matplotlib, separate accuracy and loss plots are generated, showing both training and validation trends with epochs on the x-axis.

   - **Trend Analysis**: The plots help identify performance trends, indicating generalization, overfitting, or underfitting through accuracy improvement and loss minimization.
