# Pokémon Type Classification using Convolutional Neural Networks

## Data and Purpose

### Dataset
- **Name:** Pokémon Images - First Generation (17000 files)
- **Author:** Mikołaj Kolman
- **Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/mikoajkolman/pokemon-images-first-generation17000-files)
- **Year:** 2023

### Image Content
The dataset contains approximately 17,000 images of first-generation Pokémon (151 species) in various styles, poses, and contexts. Unlike official game renders, these images include fan art, screenshots from the anime, trading cards, and other media representations, providing a diverse set of visual representations for each Pokémon.

### Problem Statement
We aim to classify Pokémon images by their elemental type (e.g., fire, water, grass) using only visual information. This task tests whether neural networks can learn the visual characteristics associated with each Pokémon type, even when the same Pokémon appears in different artistic styles and contexts. Since many Pokémon have dual typing, we focus on predicting the primary type (Type1) for this initial implementation.

## Method

### Neural Network Approach
We implement a Convolutional Neural Network (CNN) for this classification task. CNNs are particularly well-suited for image classification as they can automatically learn hierarchical features from images:
- Lower layers detect simple features like edges and textures
- Middle layers combine these into more complex patterns
- Higher layers identify type-specific visual traits (like fiery elements for Fire types)

### CNN Architecture
Our model uses a sequential architecture with:
1. Four convolutional blocks, each consisting of:
   - Convolutional layer with increasing filter counts (32 → 64 → 128 → 128)
   - ReLU activation
   - Max pooling for downsampling
2. Flatten layer to convert 2D feature maps to 1D
3. Dropout (0.5) for regularization to prevent overfitting
4. Dense layer with 512 units and ReLU activation
5. Output layer with softmax activation for multi-class classification

### Computer Vision Problem Type
This is an **image classification** problem where the model assigns a single label (Pokémon type) to an input image.

## Implementation

### Data Preprocessing and Splitting
1. We filter the dataset to include only types with at least 10 representatives (combining Type1 and Type2 occurrences).
2. We address class imbalance by:
   - Tracking images per Pokémon and per type
   - Balancing classes through oversampling (for underrepresented types) or undersampling (for overrepresented types)
3. We split the balanced dataset into:
   - 80% training set
   - 10% validation set (for hyperparameter tuning)
   - 10% test set (for final evaluation)

### Data Augmentation
To increase the effective size of our training set and improve model generalization, we apply several augmentation techniques:
- Random rotations (±20°)
- Width and height shifts (±20%)
- Shearing transformations
- Zoom variations
- Horizontal flips

### Training Process
The model is trained using:
- Adam optimizer with learning rate 1e-4
- Categorical cross-entropy loss function
- Accuracy metric for evaluation
- 20 epochs (adjustable)
- Balanced batch sampling to handle any remaining class imbalance

### Evaluation Metrics
We evaluate the model using:
- Test accuracy: percentage of correctly classified images
- Test loss: categorical cross-entropy on test set
- Per-class precision, recall, and F1-score (implied in the implementation)
- Visual evaluation of predictions on test images

### Visualizations
The implementation generates:
1. Training history plots showing:
   - Training and validation accuracy over epochs
   - Training and validation loss over epochs
2. Prediction visualizations on test images, displaying:
   - The original image
   - True type label
   - Predicted type label

### Issues and Solutions

#### Challenge 1: Class Imbalance
- **Issue:** Uneven distribution of Pokémon types (e.g., many Water types, few Dragon types)
- **Solution:** 
  - Filter out extremely rare types (<10 instances)
  - Balance classes through over/undersampling to ensure equal representation
  - Use stratified sampling for train/validation/test splits

#### Challenge 2: Image Variability
- **Issue:** High variability in image styles, backgrounds, and Pokémon poses
- **Solution:**
  - Data augmentation to help the model generalize
  - CNN architecture with sufficient depth to learn robust features
  - Using RGB images to preserve color information important for type identification

#### Challenge 3: Multi-label Nature
- **Issue:** Many Pokémon have dual types (primary and secondary)
- **Solution:**
  - Focus on primary type prediction for this initial implementation
  - Filtered dataset to ensure consistent typing information
  - Future work could extend to multi-label classification for both types

## Usage

### Training the Model
```
python pokemon_type_classifier.py
```

### Predicting Types for New Images
```
python predict_type.py <path_to_image>
```

## Future Improvements
1. Implement multi-label classification for both primary and secondary types
2. Experiment with more complex architectures (ResNet, EfficientNet)
3. Add attention mechanisms to focus on discriminative regions of Pokémon images
4. Expand to later generation Pokémon 