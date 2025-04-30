import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from collections import Counter
import warnings

# Ignore PIL warnings
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.Image")

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# %%
# Constants
IMG_SIZE = (128, 128)  # Resize images to this size
BATCH_SIZE = 32
EPOCHS = 20
DATA_DIR = "data"
METADATA_FILE = "metadata.csv"
IMAGES_PER_CLASS = 1000  # Exact number of images required per class

# %%
def main():
    print("Pokemon Type Classifier using CNN - 1000 Images Per Class Version")
    print("Loading and processing data...")
    
    # Load metadata
    metadata = pd.read_csv(METADATA_FILE)
    
    # Data exploration and analysis
    print(f"Total Pokemon: {len(metadata)}")
    
    # Create image paths and labels
    image_paths = []
    labels = []
    pokemon_to_images = {}  # Dictionary to track images per Pokemon
    
    # First pass: collect all images by Pokemon
    for _, row in metadata.iterrows():
        pokemon_name = row['Name']
        pokemon_dir = os.path.join(DATA_DIR, os.path.basename(row['Path'].strip('./')))
        
        if os.path.exists(pokemon_dir):
            # Initialize list for this Pokemon
            pokemon_to_images[pokemon_name] = []
            
            # Get all image files in the Pokemon's directory
            for img_file in os.listdir(pokemon_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    img_path = os.path.join(pokemon_dir, img_file)
                    pokemon_to_images[pokemon_name].append(img_path)
    
    # Group Pokemon by their primary type
    type_to_pokemon = {}
    for _, row in metadata.iterrows():
        type_name = row['Type1']
        pokemon_name = row['Name']
        if type_name not in type_to_pokemon:
            type_to_pokemon[type_name] = []
        type_to_pokemon[type_name].append(pokemon_name)
    
    # Calculate images per type
    type_to_image_count = {}
    for type_name, pokemon_list in type_to_pokemon.items():
        type_to_image_count[type_name] = sum([
            len(pokemon_to_images.get(pokemon, [])) for pokemon in pokemon_list
        ])
    
    print("\nTotal images per type:")
    for type_name, count in type_to_image_count.items():
        print(f"{type_name}: {count}")
    
    # Filter types that have at least IMAGES_PER_CLASS images
    valid_types = [t for t, count in type_to_image_count.items() if count >= IMAGES_PER_CLASS]
    print(f"\nTypes with at least {IMAGES_PER_CLASS} images: {valid_types}")
    
    # Create balanced dataset with exactly IMAGES_PER_CLASS images per class
    balanced_image_paths = []
    balanced_labels = []
    
    # Create a dedicated random state for image selection to ensure reproducibility
    selection_random_state = np.random.RandomState(seed=42)
    
    print("\nBalancing dataset...")
    for type_name in valid_types:
        # Get all images for this type
        type_images = []
        for pokemon in type_to_pokemon[type_name]:
            if pokemon in pokemon_to_images:
                type_images.extend(pokemon_to_images[pokemon])
        
        print(f"{type_name}: {len(type_images)} images available, selecting {IMAGES_PER_CLASS}")
        
        # Undersample to reach exactly IMAGES_PER_CLASS
        selected_images = selection_random_state.choice(
            type_images,
            size=IMAGES_PER_CLASS,
            replace=False
        )
        balanced_image_paths.extend(selected_images)
        balanced_labels.extend([type_name] * IMAGES_PER_CLASS)
    
    print(f"\nTotal balanced images: {len(balanced_image_paths)}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(balanced_labels)
    num_classes = len(label_encoder.classes_)
    
    print(f"Classes: {label_encoder.classes_}")
    
    # Check final class distribution
    label_counts = Counter(encoded_labels)
    print("\nFinal class distribution:")
    for label_id, count in label_counts.items():
        print(f"{label_encoder.inverse_transform([label_id])[0]}: {count}")
    
    # %%
    # Visualize the class distribution
    plt.figure(figsize=(12, 6))
    sns.barplot(x=label_encoder.inverse_transform(list(label_counts.keys())), 
                y=list(label_counts.values()))
    plt.title('Class Distribution')
    plt.xlabel('Pokémon Type')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('class_distribution_1000.png')
    plt.close()
    
    # %%
    # Split the dataset into training, validation, and test sets
    # First split into training and temp sets (80% train, 20% temp)
    X_train, X_temp, y_train, y_temp = train_test_split(
        balanced_image_paths, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
    )
    
    # Split temp into validation and test sets (50% validation, 50% test) 
    # This gives 80% train, 10% validation, 10% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"\nTraining set: {len(X_train)} images")
    print(f"Validation set: {len(X_val)} images")
    print(f"Test set: {len(X_test)} images")
    
    # %%
    # Data generators with augmentation for training
    train_datagen = ImageDataGenerator(
        # No rescaling here as we'll do it manually
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Only rescaling for validation and test sets
    val_test_datagen = ImageDataGenerator(
        # No rescaling here as we'll do it manually
    )
    
    # Custom data generator to load images from file paths
    def generate_from_paths_and_labels(image_paths, labels, batch_size, datagen):
        # Create a dedicated random state for batch selection
        batch_random_state = np.random.RandomState(seed=42)
        
        while True:
            # Select random indexes in batches
            indexes = batch_random_state.randint(0, len(image_paths), batch_size)
            batch_paths = [image_paths[i] for i in indexes]
            batch_labels = labels[indexes]
            
            # Load and preprocess images
            batch_images = []
            for img_path in batch_paths:
                try:
                    # Load image and convert to RGB
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize(IMG_SIZE)
                    
                    # Convert to numpy array and normalize to 0-1 range first
                    img_array = np.array(img, dtype=np.float32) / 255.0
                    
                    # Apply augmentation if it's the training datagen
                    if datagen.preprocessing_function is not None or datagen.rotation_range > 0:
                        # For training with augmentation, apply the random transforms
                        # Convert to uint8 temporarily for augmentation operations
                        img_array_aug = datagen.random_transform(img_array)
                        batch_images.append(img_array_aug)
                    else:
                        # For validation/test, just use the normalized array
                        batch_images.append(img_array)
                        
                except Exception as e:
                    print(f"Error loading image {img_path}: {str(e)}")
                    # Use a blank image as placeholder
                    batch_images.append(np.zeros((*IMG_SIZE, 3), dtype=np.float32))
            
            # Convert to numpy arrays
            batch_images = np.array(batch_images, dtype=np.float32)
            batch_labels_onehot = tf.keras.utils.to_categorical(batch_labels, num_classes)
            
            yield batch_images, batch_labels_onehot
    
    # Create generators
    train_generator = generate_from_paths_and_labels(
        X_train, np.array(y_train), BATCH_SIZE, train_datagen
    )
    
    val_generator = generate_from_paths_and_labels(
        X_val, np.array(y_val), BATCH_SIZE, val_test_datagen
    )
    
    test_generator = generate_from_paths_and_labels(
        X_test, np.array(y_test), BATCH_SIZE, val_test_datagen
    )
    
    # %%
    # Build the CNN model
    model = build_model(num_classes)
    
    # Compile the model
    model.compile(
        optimizer=optimizers.Adam(1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # %%
    # Train the model
    print("\nTraining the model...")
    
    # Calculate steps per epoch
    steps_per_epoch = len(X_train) // BATCH_SIZE
    validation_steps = len(X_val) // BATCH_SIZE
    
    # Ensure at least 1 step per epoch
    steps_per_epoch = max(1, steps_per_epoch)
    validation_steps = max(1, validation_steps)
    
    # Early stopping callback
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        min_delta=0.01,
        restore_best_weights=True,
        verbose=1
    )
    
    # Model checkpoint callback to save the best model
    model_checkpoint = callbacks.ModelCheckpoint(
        'best_model_1000.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=val_generator,
        validation_steps=validation_steps,
        callbacks=[early_stopping, model_checkpoint]
    )
    
    # %%
    # Evaluate the model
    print("\nEvaluating the model...")
    test_steps = max(1, len(X_test) // BATCH_SIZE)
    test_loss, test_acc = model.evaluate(test_generator, steps=test_steps)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Plot training history
    plot_training_history(history)
    
    # Generate detailed metrics and confusion matrix
    y_pred = get_predictions(model, X_test, val_test_datagen)
    generate_metrics(y_test, y_pred, label_encoder)
    
    # Save the model
    model.save("pokemon_type_model_1000.keras")
    print("Model saved as pokemon_type_model_1000.keras")
    
    # Save the label encoder classes for later use
    np.save("label_encoder_classes_1000.npy", label_encoder.classes_)
    
    # Save the list of valid types
    np.save("valid_types_1000.npy", np.array(valid_types))
    
    # Display some predictions on test images
    display_predictions(model, X_test, y_test, label_encoder)

# %%
def build_model(num_classes):
    """Build a CNN model for Pokemon type classification"""
    model = models.Sequential()
    
    # First convolutional block
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*IMG_SIZE, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Second convolutional block
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Third convolutional block
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Fourth convolutional block
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Flatten and dense layers
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))  # Dropout for regularization
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    # Print model summary
    model.summary()
    
    return model

# %%
def plot_training_history(history):
    """Plot training and validation accuracy/loss"""
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.tight_layout()
    plt.savefig('training_history_1000.png')
    plt.close()

# %%
def get_predictions(model, X_test, datagen):
    """Get predictions for all test images"""
    y_pred = []
    batch_size = 32
    
    # Process test images in batches to avoid memory issues
    for i in range(0, len(X_test), batch_size):
        batch_paths = X_test[i:i+batch_size]
        batch_images = []
        
        for img_path in batch_paths:
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize(IMG_SIZE)
                img_array = np.array(img, dtype=np.float32) / 255.0
                batch_images.append(img_array)
            except Exception as e:
                print(f"Error loading image {img_path}: {str(e)}")
                batch_images.append(np.zeros((*IMG_SIZE, 3), dtype=np.float32))
        
        batch_images = np.array(batch_images, dtype=np.float32)
        batch_preds = model.predict(batch_images)
        batch_preds = np.argmax(batch_preds, axis=1)
        y_pred.extend(batch_preds)
    
    return np.array(y_pred)

# %%
def generate_metrics(y_test, y_pred, label_encoder):
    """Generate and save detailed metrics and confusion matrix"""
    # Generate classification report
    class_names = label_encoder.classes_
    report = classification_report(
        y_test, y_pred, 
        target_names=class_names, 
        output_dict=True
    )
    
    # Convert report to DataFrame for easier manipulation and saving
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv('classification_report_1000.csv')
    
    # Print report to console
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig('confusion_matrix_1000.png')
    plt.close()
    
    # Calculate and print overall metrics
    accuracy = np.sum(y_test == y_pred) / len(y_test)
    print(f"\nOverall Test Accuracy: {accuracy:.4f}")
    
    # Per-class metrics
    plt.figure(figsize=(12, 8))
    metrics_df = pd.DataFrame({
        'Precision': [report[c]['precision'] for c in class_names],
        'Recall': [report[c]['recall'] for c in class_names],
        'F1-Score': [report[c]['f1-score'] for c in class_names]
    }, index=class_names)
    
    metrics_df.plot(kind='bar', figsize=(15, 8))
    plt.title('Per-Class Performance Metrics')
    plt.xlabel('Pokemon Type')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('per_class_metrics_1000.png')
    plt.close()

# %%
def display_predictions(model, X_test, y_test, label_encoder):
    """Display predictions for a few test images"""
    # Select a few random test images
    num_samples = min(10, len(X_test))
    # Use a fixed random state for reproducible image selection
    viz_random_state = np.random.RandomState(seed=42)
    sample_indices = viz_random_state.choice(len(X_test), num_samples, replace=False)
    
    plt.figure(figsize=(15, 10))
    for i, idx in enumerate(sample_indices):
        img_path = X_test[idx]
        true_label = y_test[idx]
        
        try:
            # Load and preprocess the image
            img = Image.open(img_path).convert('RGB')
            img = img.resize(IMG_SIZE)
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_batch = np.expand_dims(img_array, axis=0)
            
            # Make prediction
            prediction = model.predict(img_batch)[0]
            predicted_label = np.argmax(prediction)
            
            # Get label names
            true_label_name = label_encoder.inverse_transform([true_label])[0]
            pred_label_name = label_encoder.inverse_transform([predicted_label])[0]
            
            # Display the image with prediction
            plt.subplot(2, 5, i + 1)
            plt.imshow(img_array)
            plt.title(f"True: {true_label_name}\nPred: {pred_label_name}")
            plt.axis('off')
        except Exception as e:
            print(f"Error displaying prediction for {img_path}: {str(e)}")
    
    plt.tight_layout()
    plt.savefig('test_predictions_1000.png')
    plt.close()

# %%
if __name__ == "__main__":
    main() 