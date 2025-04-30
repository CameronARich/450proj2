import os
import sys
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
import argparse
import datetime

# Ignore PIL warnings
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.Image")

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Constants
IMG_SIZE = (128, 128)  # Resize images to this size
BATCH_SIZE = 32
EPOCHS = 20
DATA_DIR = "data"
METADATA_FILE = "metadata.csv"
IMAGES_PER_CLASS = 1000  # Exact number of images required per class
OUTPUT_DIR = "output"  # Directory for all outputs

def create_output_dir(output_dir=None):
    """Create output directory if it doesn't exist and return path"""
    if output_dir is None:
        # Create a timestamped output directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(OUTPUT_DIR, f"run_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"All outputs will be saved to: {output_dir}")
    return output_dir

def train_model(output_dir=None):
    """Train the Pokemon type classifier model using 1000 images per class"""
    # Create output directory
    output_dir = create_output_dir(output_dir)
    
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
    
    # Visualize the class distribution
    plt.figure(figsize=(12, 6))
    sns.barplot(x=label_encoder.inverse_transform(list(label_counts.keys())), 
                y=list(label_counts.values()))
    plt.title('Class Distribution')
    plt.xlabel('Pokémon Type')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'))
    plt.close()
    
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
    
    # Save the train/val/test split information
    split_info = {
        'train': X_train,
        'val': X_val,
        'test': X_test
    }
    with open(os.path.join(output_dir, 'data_split.txt'), 'w') as f:
        for split_name, split_data in split_info.items():
            f.write(f"{split_name}: {len(split_data)} examples\n")
            
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
    
    # Build the CNN model
    model = build_model(num_classes)
    
    # Compile the model
    model.compile(
        optimizer=optimizers.Adam(1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
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
        os.path.join(output_dir, 'best_model.keras'),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    # TensorBoard callback for visualization
    tensorboard_callback = callbacks.TensorBoard(
        log_dir=os.path.join(output_dir, 'logs'),
        histogram_freq=1
    )
    
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=val_generator,
        validation_steps=validation_steps,
        callbacks=[early_stopping, model_checkpoint, tensorboard_callback]
    )
    
    # Evaluate the model
    print("\nEvaluating the model...")
    test_steps = max(1, len(X_test) // BATCH_SIZE)
    test_loss, test_acc = model.evaluate(test_generator, steps=test_steps)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Save evaluation results
    with open(os.path.join(output_dir, 'evaluation_results.txt'), 'w') as f:
        f.write(f"Test accuracy: {test_acc:.4f}\n")
        f.write(f"Test loss: {test_loss:.4f}\n")
    
    # Plot training history
    plot_training_history(history, output_dir)
    
    # Generate detailed metrics and confusion matrix
    y_pred, _ = get_predictions(model, X_test, val_test_datagen)
    generate_metrics(y_test, y_pred, label_encoder, output_dir)
    
    # Save the model
    model.save(os.path.join(output_dir, "model.keras"))
    print(f"Model saved to {os.path.join(output_dir, 'model.keras')}")
    
    # Save the label encoder classes for later use
    np.save(os.path.join(output_dir, "label_encoder_classes.npy"), label_encoder.classes_)
    
    # Save the list of valid types
    np.save(os.path.join(output_dir, "valid_types.npy"), np.array(valid_types))
    
    # Display some predictions on test images
    display_predictions(model, X_test, y_test, label_encoder, output_dir)
    
    # Save test predictions with logits
    save_test_predictions(model, X_test, y_test, label_encoder, output_dir)
    
    return output_dir


def build_model(num_classes):
    """Build a CNN model for Pokemon type classification"""
    # Use the functional API with Input layer as recommended
    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    
    # First convolutional block
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Second convolutional block
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Third convolutional block
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Fourth convolutional block
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Flatten and dense layers
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)  # Dropout for regularization
    x = layers.Dense(512, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Print model summary
    model.summary()
    
    return model


def plot_training_history(history, output_dir):
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
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()
    
    # Also save the history data as CSV
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(output_dir, 'training_history.csv'), index=False)


def get_predictions(model, X_test, datagen):
    """Get predictions for all test images"""
    y_pred = []
    all_probs = []  # Store all probabilities
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
        batch_probs = model.predict(batch_images)  # Get probabilities
        all_probs.extend(batch_probs)  # Store all probabilities
        batch_preds = np.argmax(batch_probs, axis=1)  # Get predicted classes
        y_pred.extend(batch_preds)
    
    return np.array(y_pred), np.array(all_probs)


def generate_metrics(y_test, y_pred, label_encoder, output_dir):
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
    report_df.to_csv(os.path.join(output_dir, 'classification_report.csv'))
    
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
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
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
    plt.savefig(os.path.join(output_dir, 'per_class_metrics.png'))
    plt.close()


def display_predictions(model, X_test, y_test, label_encoder, output_dir):
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
    plt.savefig(os.path.join(output_dir, 'test_predictions.png'))
    plt.close()


def save_test_predictions(model, X_test, y_test, label_encoder, output_dir):
    """Save predictions with logits for all test examples"""
    print("\nGenerating and saving prediction logits for all test examples...")
    
    # Get predictions and probabilities
    y_pred, all_probs = get_predictions(model, X_test, None)
    
    # Create a DataFrame to store results
    results = []
    
    for i, (img_path, true_label, pred_label, probs) in enumerate(zip(X_test, y_test, y_pred, all_probs)):
        # Get label names
        true_label_name = label_encoder.inverse_transform([true_label])[0]
        pred_label_name = label_encoder.inverse_transform([pred_label])[0]
        
        # Create a row with basic info
        row = {
            'example_id': i,
            'image_path': img_path,
            'true_label': true_label_name,
            'predicted_label': pred_label_name,
            'correct': true_label == pred_label,
        }
        
        # Add probabilities for each class
        for j, class_name in enumerate(label_encoder.classes_):
            row[f'prob_{class_name}'] = probs[j]
        
        results.append(row)
    
    # Convert to DataFrame and save
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'test_predictions_with_logits.csv'), index=False)
    print(f"Saved predictions with logits for {len(results_df)} test examples")
    
    # Also save a version with just the top predictions for quick analysis
    top_results = []
    for i, row in enumerate(results):
        probs = all_probs[i]
        # Get top 3 predictions
        top_indices = np.argsort(probs)[::-1][:3]
        
        top_row = {
            'example_id': i,
            'image_path': row['image_path'],
            'true_label': row['true_label'],
            'correct': row['correct']
        }
        
        # Add top 3 predictions with probabilities
        for rank, idx in enumerate(top_indices):
            class_name = label_encoder.classes_[idx]
            prob = probs[idx]
            top_row[f'pred{rank+1}'] = class_name
            top_row[f'prob{rank+1}'] = prob
            
        top_results.append(top_row)
    
    # Save top predictions
    top_df = pd.DataFrame(top_results)
    top_df.to_csv(os.path.join(output_dir, 'test_predictions_top3.csv'), index=False)
    print(f"Saved top 3 predictions for {len(top_df)} test examples")
    
    return results_df


def predict_single_image(image_path, show_image=True, top_k=5, output_dir=None):
    """Predict the type of a single Pokemon image and show all logits/probabilities"""
    # Create output directory if specified or needed
    if output_dir is None:
        output_dir = create_output_dir()
    
    # Check which model file to use
    model_paths = [
        "model.keras",
        "best_model.keras",
        "pokemon_type_model_1000.keras",
        "best_model_1000.keras"
    ]
    
    model_path = None
    for path in model_paths:
        # Try both in current directory and in output dir
        if os.path.exists(path):
            model_path = path
            break
        elif os.path.exists(os.path.join(output_dir, path)):
            model_path = os.path.join(output_dir, path)
            break
    
    if model_path is None:
        print("Error: No model file found. Please train the model first.")
        return None, None
    
    print(f"Using model: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    # Find class names file
    class_file_paths = [
        "label_encoder_classes.npy",
        "label_encoder_classes_1000.npy"
    ]
    
    class_file = None
    for path in class_file_paths:
        # Try both in current directory and in output dir
        if os.path.exists(path):
            class_file = path
            break
        elif os.path.exists(os.path.join(output_dir, path)):
            class_file = os.path.join(output_dir, path)
            break
    
    if class_file is None:
        print("Error: No class file found. Please train the model first.")
        return None, None
    
    print(f"Using classes from: {class_file}")
    classes = np.load(class_file, allow_pickle=True)
    
    try:
        # Load and preprocess the image
        img = Image.open(image_path).convert('RGB')
        img = img.resize(IMG_SIZE)
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)
        
        # Get prediction probabilities
        prediction_probs = model.predict(img_batch)[0]
        
        # Sort indices by probability (highest first)
        sorted_indices = np.argsort(prediction_probs)[::-1]
        
        # Display the image
        if show_image:
            plt.figure(figsize=(10, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(img_array)
            plt.axis('off')
            plt.title("Input Image")
            
            # Create a horizontal bar chart for probabilities
            plt.subplot(1, 2, 2)
            
            # Select top k predictions
            top_indices = sorted_indices[:top_k]
            top_probs = prediction_probs[top_indices]
            top_classes = classes[top_indices]
            
            # Horizontal bar chart
            y_pos = np.arange(len(top_indices))
            plt.barh(y_pos, top_probs, align='center')
            plt.yticks(y_pos, top_classes)
            plt.xlabel('Probability')
            plt.title('Top Predictions')
            
            plt.tight_layout()
            prediction_viz_path = os.path.join(output_dir, 'prediction_probs.png')
            plt.savefig(prediction_viz_path)
            print(f"Visualization saved to {prediction_viz_path}")
            plt.show()
        
        # Print all probabilities
        print(f"\nPredictions for {os.path.basename(image_path)}:")
        print(f"{'Type':<15} {'Probability':<15} {'Confidence'}")
        print("-" * 45)
        
        # Save detailed predictions to CSV
        pred_results = []
        for i in range(len(classes)):
            idx = sorted_indices[i]
            class_name = classes[idx]
            prob = prediction_probs[idx]
            stars = int(prob * 50)  # Visual confidence indicator
            confidence = '*' * stars
            print(f"{class_name:<15} {prob:.6f} {confidence}")
            
            pred_results.append({
                'type': class_name,
                'probability': prob
            })
        
        # Save to CSV
        pred_df = pd.DataFrame(pred_results)
        pred_csv_path = os.path.join(output_dir, 'single_prediction_results.csv')
        pred_df.to_csv(pred_csv_path, index=False)
        print(f"Prediction results saved to {pred_csv_path}")
        
        return classes, prediction_probs
        
    except Exception as e:
        print(f"Error predicting image {image_path}: {str(e)}")
        return None, None


def main():
    """Main function to handle command line arguments"""
    parser = argparse.ArgumentParser(description='Pokemon Type Classifier')
    
    # Create subparsers for train and predict commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--output-dir', type=str, help='Custom output directory')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict the type of a Pokemon image')
    predict_parser.add_argument('image_path', type=str, help='Path to the image')
    predict_parser.add_argument('--no-image', action='store_true', help='Do not show the image visualization')
    predict_parser.add_argument('--top', type=int, default=5, help='Number of top predictions to visualize')
    predict_parser.add_argument('--output-dir', type=str, help='Custom output directory')
    
    args = parser.parse_args()
    
    # Default to train if no command is provided
    if args.command == 'predict':
        predict_single_image(
            args.image_path, 
            not args.no_image, 
            args.top,
            args.output_dir
        )
    else:
        # Default to training mode (including when no command is specified)
        print("No command specified or 'train' command used. Running in training mode...")
        output_dir = args.output_dir if hasattr(args, 'output_dir') else None
        train_model(output_dir)


if __name__ == "__main__":
    main() 