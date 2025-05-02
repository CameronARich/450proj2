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
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    multilabel_confusion_matrix,
)
import seaborn as sns
from collections import Counter
import warnings
import datetime
import glob

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
OUTPUT_DIR = "output"  # Directory for all outputs
MIN_IMAGES_PER_CLASS = 1000  # Minimum number of images required per class
MAX_IMAGES_PER_CLASS = 2000  # Maximum number of images per class


def create_output_dir(output_dir=None):
    """Create output directory if it doesn't exist and return path"""
    if output_dir is None:
        # Create a timestamped output directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(OUTPUT_DIR, f"run_{timestamp}")

    os.makedirs(output_dir, exist_ok=True)
    print(f"All outputs will be saved to: {output_dir}")
    return output_dir


def load_metadata():
    """Load and process metadata file"""
    metadata = pd.read_csv(METADATA_FILE)
    print(f"Total Pokemon: {len(metadata)}")
    return metadata


def extract_type_information(metadata):
    """Extract unique types and type information from metadata"""
    all_types = set()
    pokemon_to_types = {}  # Dictionary to map Pokemon to their types

    for _, row in metadata.iterrows():
        pokemon_name = row["Name"]
        type1 = row["Type1"]
        type2 = row["Type2"] if pd.notna(row["Type2"]) and row["Type2"] != "" else None

        # Add to set of all types
        all_types.add(type1)
        if type2:
            all_types.add(type2)

        # Record types for each Pokemon
        pokemon_to_types[pokemon_name] = [type1] if type2 is None else [type1, type2]

    all_types = sorted(list(all_types))
    print(f"\nTotal unique types: {len(all_types)}")
    print(f"Types: {all_types}")
    
    return all_types, pokemon_to_types


def collect_pokemon_images(metadata):
    """Collect all images for each Pokemon"""
    pokemon_to_images = {}  # Dictionary to track images per Pokemon

    for _, row in metadata.iterrows():
        pokemon_name = row["Name"]
        pokemon_dir = os.path.join(DATA_DIR, os.path.basename(row["Path"].strip("./")))

        if os.path.exists(pokemon_dir):
            # Initialize list for this Pokemon
            pokemon_to_images[pokemon_name] = []

            # Get all image files in the Pokemon's directory
            for img_file in os.listdir(pokemon_dir):
                if img_file.lower().endswith((".png", ".jpg", ".jpeg", ".gif")):
                    img_path = os.path.join(pokemon_dir, img_file)
                    pokemon_to_images[pokemon_name].append(img_path)
    
    return pokemon_to_images


def count_images_per_type(all_types, pokemon_to_types, pokemon_to_images):
    """Count images per type"""
    type_to_image_count = {t: 0 for t in all_types}
    
    for pokemon, types in pokemon_to_types.items():
        if pokemon in pokemon_to_images:
            img_count = len(pokemon_to_images[pokemon])
            for t in types:
                type_to_image_count[t] += img_count

    print("\nTotal images per type:")
    for type_name, count in sorted(type_to_image_count.items()):
        print(f"{type_name}: {count}")
        
    return type_to_image_count


def filter_valid_types(all_types, type_to_image_count, min_images=MIN_IMAGES_PER_CLASS):
    """Filter types with sufficient images"""
    valid_types = [
        t for t, count in type_to_image_count.items() if count >= min_images
    ]
    print(f"\nTypes with at least {min_images} images: {valid_types}")
    return valid_types


def determine_target_counts(valid_types, type_to_image_count, min_images=MIN_IMAGES_PER_CLASS, max_images=MAX_IMAGES_PER_CLASS):
    """Determine target number of images per type"""
    type_to_target_count = {}
    
    for t in valid_types:
        # Calculate a target that scales between MIN and MAX based on available images
        available = type_to_image_count[t]
        if available <= max_images:
            # If we have fewer than MAX, use all available but at least MIN
            type_to_target_count[t] = max(min_images, available)
        else:
            # If we have more than MAX, cap at MAX
            type_to_target_count[t] = max_images
    
    print("\nTarget images per type:")
    for type_name, count in sorted(type_to_target_count.items()):
        print(f"{type_name}: {count}")
        
    return type_to_target_count


def group_pokemon_by_type(valid_types, pokemon_to_types, pokemon_to_images):
    """Group Pokemon by their types"""
    type_to_pokemon = {t: [] for t in valid_types}
    
    for pokemon, types in pokemon_to_types.items():
        if pokemon in pokemon_to_images and len(pokemon_to_images[pokemon]) > 0:
            for t in types:
                if t in valid_types:
                    type_to_pokemon[t].append(pokemon)
                    
    return type_to_pokemon


def calculate_pokemon_allocations(valid_types, type_to_pokemon, type_to_target_count, pokemon_to_images):
    """Calculate how many images to select from each Pokemon for each type"""
    pokemon_type_to_target_images = {}
    
    for t in valid_types:
        target_count = type_to_target_count[t]
        pokemon_list = type_to_pokemon[t]
        
        # Skip empty types
        if not pokemon_list:
            continue
            
        # Calculate approximately how many images to take from each Pokemon
        # Distribute target count proportionally across Pokemon
        pokemon_counts = {p: len(pokemon_to_images[p]) for p in pokemon_list}
        total_available = sum(pokemon_counts.values())
        
        # Calculate allocation proportionally but with a minimum
        pokemon_allocations = {}
        for p in pokemon_list:
            # Base allocation is proportional to available images
            proportion = pokemon_counts[p] / total_available
            allocation = int(proportion * target_count)
            
            # Ensure minimum of 1 image if any are available
            allocation = max(1, allocation)
            
            # But never more than available
            allocation = min(allocation, pokemon_counts[p])
            
            pokemon_allocations[p] = allocation
        
        # Adjust allocations to match target count as closely as possible
        total_allocated = sum(pokemon_allocations.values())
        
        # If we're under target, try to allocate more
        if total_allocated < target_count:
            # Sort Pokemon by how many more they could contribute
            remaining_capacity = {
                p: pokemon_counts[p] - pokemon_allocations[p] 
                for p in pokemon_list
            }
            
            # Sort by remaining capacity, descending
            sorted_pokemon = sorted(
                pokemon_list, key=lambda p: remaining_capacity[p], reverse=True
            )
            
            # Allocate additional images until we reach target
            for p in sorted_pokemon:
                if total_allocated >= target_count:
                    break
                    
                available_to_add = remaining_capacity[p]
                to_add = min(available_to_add, target_count - total_allocated)
                
                if to_add > 0:
                    pokemon_allocations[p] += to_add
                    total_allocated += to_add
        
        # If we're over target, reduce allocations
        elif total_allocated > target_count:
            # Sort by current allocation, descending
            sorted_pokemon = sorted(
                pokemon_list, key=lambda p: pokemon_allocations[p], reverse=True
            )
            
            # Reduce allocations proportionally
            for p in sorted_pokemon:
                if total_allocated <= target_count:
                    break
                    
                # Don't reduce below 1
                if pokemon_allocations[p] > 1:
                    to_reduce = min(pokemon_allocations[p] - 1, total_allocated - target_count)
                    pokemon_allocations[p] -= to_reduce
                    total_allocated -= to_reduce
        
        # Store the final allocations
        for p in pokemon_list:
            pokemon_type_to_target_images[(p, t)] = pokemon_allocations[p]
            
    return pokemon_type_to_target_images


def create_balanced_dataset(valid_types, pokemon_to_types, pokemon_to_images, pokemon_type_to_target_images, type_to_target_count):
    """Create a balanced dataset with appropriate image selection"""
    # Create a dedicated random state for image selection to ensure reproducibility
    selection_random_state = np.random.RandomState(seed=42)
    
    pokemon_to_selected_images = {p: 0 for p in pokemon_to_images.keys()}
    type_to_selected_images = {t: 0 for t in valid_types}
    selected_image_paths = set()
    balanced_image_paths = []
    balanced_labels = []
    
    # First, handle Pokemon that contribute to only one type
    for pokemon, types in pokemon_to_types.items():
        valid_types_for_pokemon = [t for t in types if t in valid_types]
        
        # Skip Pokemon with no valid types or no images
        if not valid_types_for_pokemon or pokemon not in pokemon_to_images:
            continue
            
        # Handle single-type Pokemon first
        if len(valid_types_for_pokemon) == 1:
            type_name = valid_types_for_pokemon[0]
            target_count = pokemon_type_to_target_images.get((pokemon, type_name), 0)
            
            if target_count == 0:
                continue
                
            available_images = pokemon_to_images[pokemon]
            
            # Skip if no images available
            if not available_images:
                continue
                
            # Select up to target_count images
            num_to_select = min(target_count, len(available_images))
            selected_images = selection_random_state.choice(
                available_images, 
                size=num_to_select, 
                replace=False
            )
            
            # Add to our balanced dataset
            for img_path in selected_images:
                if img_path not in selected_image_paths:
                    selected_image_paths.add(img_path)
                    balanced_image_paths.append(img_path)
                    balanced_labels.append(valid_types_for_pokemon)
                    pokemon_to_selected_images[pokemon] += 1
                    type_to_selected_images[type_name] += 1
    
    # Now handle dual-type Pokemon
    for pokemon, types in pokemon_to_types.items():
        valid_types_for_pokemon = [t for t in types if t in valid_types]
        
        # Skip Pokemon with no valid types, no images, or single type (already handled)
        if (not valid_types_for_pokemon or pokemon not in pokemon_to_images 
            or len(valid_types_for_pokemon) != 2):
            continue
            
        # For dual-type, determine target based on both types
        type1, type2 = valid_types_for_pokemon
        target1 = pokemon_type_to_target_images.get((pokemon, type1), 0)
        target2 = pokemon_type_to_target_images.get((pokemon, type2), 0)
        
        # Take the average as the initial target
        target_count = max(1, (target1 + target2) // 2)
        
        # But adjust if a type is already near its overall target
        remaining_type1 = type_to_target_count.get(type1, 0) - type_to_selected_images[type1]
        remaining_type2 = type_to_target_count.get(type2, 0) - type_to_selected_images[type2]
        
        # Cap target based on remaining needs
        target_count = min(target_count, remaining_type1, remaining_type2)
        
        # Skip if no images needed
        if target_count <= 0:
            continue
            
        available_images = [
            img for img in pokemon_to_images[pokemon] 
            if img not in selected_image_paths
        ]
        
        # Skip if no images available
        if not available_images:
            continue
            
        # Select up to target_count images
        num_to_select = min(target_count, len(available_images))
        selected_images = selection_random_state.choice(
            available_images, 
            size=num_to_select, 
            replace=False
        )
        
        # Add to our balanced dataset
        for img_path in selected_images:
            selected_image_paths.add(img_path)
            balanced_image_paths.append(img_path)
            balanced_labels.append(valid_types_for_pokemon)
            pokemon_to_selected_images[pokemon] += 1
            type_to_selected_images[type1] += 1
            type_to_selected_images[type2] += 1
            
    return balanced_image_paths, balanced_labels, type_to_selected_images, pokemon_to_selected_images


def ensure_minimum_counts(valid_types, type_to_selected_images, pokemon_to_images, pokemon_to_types, selected_image_paths, balanced_image_paths, balanced_labels, type_to_pokemon, type_to_target_count, min_images=MIN_IMAGES_PER_CLASS):
    """Ensure minimum image counts for each type"""
    selection_random_state = np.random.RandomState(seed=42)
    
    for type_name in valid_types:
        if type_to_selected_images[type_name] < min_images:
            # We need more images for this type
            shortage = min_images - type_to_selected_images[type_name]
            print(f"Need {shortage} more images for {type_name}")
            
            # Get all Pokemon of this type
            pokemon_list = type_to_pokemon[type_name]
            
            # Try to find more images from these Pokemon
            for pokemon in pokemon_list:
                if shortage <= 0:
                    break
                    
                # Get images we haven't selected yet
                available_images = [
                    img for img in pokemon_to_images[pokemon]
                    if img not in selected_image_paths
                ]
                
                if not available_images:
                    continue
                    
                # Select up to the shortage count
                num_to_select = min(shortage, len(available_images))
                selected_images = selection_random_state.choice(
                    available_images,
                    size=num_to_select,
                    replace=False
                )
                
                # Add to our dataset
                for img_path in selected_images:
                    selected_image_paths.add(img_path)
                    balanced_image_paths.append(img_path)
                    # Get all valid types for this Pokemon
                    pokemon_types = [t for t in pokemon_to_types[pokemon] if t in valid_types]
                    balanced_labels.append(pokemon_types)
                    # Update counts for all types of this Pokemon
                    for t in pokemon_types:
                        type_to_selected_images[t] += 1
                
                shortage -= num_to_select
                
    return balanced_image_paths, balanced_labels, type_to_selected_images


def create_data_generators():
    """Create data generators with appropriate augmentation"""
    # Data generators with augmentation for training
    train_datagen = ImageDataGenerator(
        # No rescaling here as we'll do it manually
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    # Only rescaling for validation and test sets
    val_test_datagen = ImageDataGenerator()
    
    return train_datagen, val_test_datagen


def load_and_preprocess_image(img_path, datagen=None):
    """Load, resize, and preprocess a single image"""
    try:
        # Load image and convert to RGB
        img = Image.open(img_path).convert("RGB")
        img = img.resize(IMG_SIZE)

        # Convert to numpy array and normalize to 0-1 range
        img_array = np.array(img, dtype=np.float32) / 255.0

        # Apply augmentation if datagen is provided with transformations
        if datagen is not None and (datagen.preprocessing_function is not None or datagen.rotation_range > 0):
            img_array = datagen.random_transform(img_array)
            
        return img_array
    except Exception as e:
        print(f"Error loading image {img_path}: {str(e)}")
        # Return a blank image as placeholder
        return np.zeros((*IMG_SIZE, 3), dtype=np.float32)


def generate_from_paths_and_labels(image_paths, labels, batch_size, datagen):
    """Custom generator to create batches from image paths and labels"""
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
            img_array = load_and_preprocess_image(img_path, datagen)
            batch_images.append(img_array)

        # Convert to numpy arrays
        batch_images = np.array(batch_images, dtype=np.float32)

        # Yield batch
        yield batch_images, batch_labels


def prepare_training_data(X_train, y_train, X_val, y_val, X_test, y_test):
    """Prepare data generators for training, validation and testing"""
    # Create data generators
    train_datagen, val_test_datagen = create_data_generators()
    
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
    
    return train_generator, val_generator, test_generator, train_datagen, val_test_datagen


def create_callbacks(output_dir):
    """Create callbacks for model training"""
    # Early stopping callback
    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        min_delta=0.01,
        restore_best_weights=True,
        verbose=1,
    )

    # Model checkpoint callback to save the best model
    model_checkpoint = callbacks.ModelCheckpoint(
        os.path.join(output_dir, "best_model.keras"),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )

    # TensorBoard callback for visualization
    tensorboard_callback = callbacks.TensorBoard(
        log_dir=os.path.join(output_dir, "logs"), histogram_freq=1
    )
    
    return [early_stopping, model_checkpoint, tensorboard_callback]


def train_model(output_dir=None):
    """Train the Pokemon type classifier model using multilabel classification"""
    # Create output directory
    output_dir = create_output_dir(output_dir)

    print("Pokemon Type Classifier using CNN - Multilabel Classification")
    print("Loading and processing data...")

    # Load metadata
    metadata = load_metadata()

    # Extract type information
    all_types, pokemon_to_types = extract_type_information(metadata)
    
    # Collect all images by Pokemon
    pokemon_to_images = collect_pokemon_images(metadata)
    
    # Count images per type
    type_to_image_count = count_images_per_type(all_types, pokemon_to_types, pokemon_to_images)
    
    # Filter valid types
    valid_types = filter_valid_types(all_types, type_to_image_count)
    
    # Determine target counts per type
    type_to_target_count = determine_target_counts(valid_types, type_to_image_count)
    
    # Group Pokemon by type
    type_to_pokemon = group_pokemon_by_type(valid_types, pokemon_to_types, pokemon_to_images)
    
    # Calculate allocation of images per Pokemon per type
    pokemon_type_to_target_images = calculate_pokemon_allocations(
        valid_types, type_to_pokemon, type_to_target_count, pokemon_to_images
    )
    
    print("\nBalancing dataset...")
    
    # Create balanced dataset
    balanced_image_paths, balanced_labels, type_to_selected_images, pokemon_to_selected_images = create_balanced_dataset(
        valid_types, pokemon_to_types, pokemon_to_images, pokemon_type_to_target_images, type_to_target_count
    )
    
    # Ensure minimum counts for each type
    balanced_image_paths, balanced_labels, type_to_selected_images = ensure_minimum_counts(
        valid_types, type_to_selected_images, pokemon_to_images, pokemon_to_types, 
        set(balanced_image_paths), balanced_image_paths, balanced_labels, 
        type_to_pokemon, type_to_target_count
    )
    
    print(f"\nTotal balanced images: {len(balanced_image_paths)}")
    
    # Print final type distribution
    print("\nFinal type distribution:")
    for type_name, count in sorted(type_to_selected_images.items()):
        print(f"{type_name}: {count}")
        
    # Print Pokemon distribution stats
    pokemon_counts = Counter(v for k, v in pokemon_to_selected_images.items() if v > 0)
    print("\nPokemon image count distribution:")
    for count, num_pokemon in sorted(pokemon_counts.items()):
        if count > 0:  # Only show Pokemon that have images
            print(f"{count} images: {num_pokemon} Pokemon")

    # Multi-hot encode the labels
    mlb = MultiLabelBinarizer(classes=valid_types)
    encoded_labels = mlb.fit_transform(balanced_labels)
    print(f"Label shape: {encoded_labels.shape}")

    # Check final class distribution
    type_counts = encoded_labels.sum(axis=0)
    print("\nFinal class distribution:")
    for i, count in enumerate(type_counts):
        print(f"{valid_types[i]}: {count}")

    # Visualize the class distribution
    plt.figure(figsize=(12, 6))
    sns.barplot(x=valid_types, y=type_counts)
    plt.title("Type Distribution")
    plt.xlabel("Pokémon Type")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "class_distribution.png"))
    plt.close()

    # Split the dataset into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        balanced_image_paths,
        encoded_labels,
        test_size=0.2,
        random_state=42,
        stratify=None,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=None
    )

    print(f"\nTraining set: {len(X_train)} images")
    print(f"Validation set: {len(X_val)} images")
    print(f"Test set: {len(X_test)} images")

    # Save the train/val/test split information
    split_info = {"train": X_train, "val": X_val, "test": X_test}
    with open(os.path.join(output_dir, "data_split.txt"), "w") as f:
        for split_name, split_data in split_info.items():
            f.write(f"{split_name}: {len(split_data)} examples\n")

    # Prepare data generators
    train_generator, val_generator, test_generator, _, val_test_datagen = prepare_training_data(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    # Build the CNN model
    num_classes = len(valid_types)
    model = build_model(num_classes)

    # Compile the model with binary crossentropy for multilabel
    model.compile(
        optimizer=optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    # Train the model
    print("\nTraining the model...")

    # Calculate steps per epoch
    steps_per_epoch = max(1, len(X_train) // BATCH_SIZE)
    validation_steps = max(1, len(X_val) // BATCH_SIZE)

    # Create callbacks
    model_callbacks = create_callbacks(output_dir)

    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=val_generator,
        validation_steps=validation_steps,
        callbacks=model_callbacks,
    )

    # Evaluate the model
    print("\nEvaluating the model...")
    test_steps = max(1, len(X_test) // BATCH_SIZE)
    test_loss, test_acc = model.evaluate(test_generator, steps=test_steps)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    # Save evaluation results
    with open(os.path.join(output_dir, "evaluation_results.txt"), "w") as f:
        f.write(f"Test accuracy: {test_acc:.4f}\n")
        f.write(f"Test loss: {test_loss:.4f}\n")

    # Plot training history
    plot_training_history(history, output_dir)

    # Generate detailed metrics and confusion matrix
    y_pred, y_probs = get_predictions(model, X_test, val_test_datagen)
    generate_metrics(y_test, y_pred, mlb, output_dir)

    # Save the model
    model.save(os.path.join(output_dir, "model.keras"))
    print(f"Model saved to {os.path.join(output_dir, 'model.keras')}")

    # Save the label encoder classes for later use
    np.save(os.path.join(output_dir, "label_encoder_classes.npy"), mlb.classes_)

    # Save the list of valid types
    np.save(os.path.join(output_dir, "valid_types.npy"), np.array(valid_types))

    # Display some predictions on test images
    display_predictions(model, X_test, y_test, mlb, output_dir)

    # Save test predictions with logits
    save_test_predictions(model, X_test, y_test, mlb, output_dir)

    return output_dir


def build_model(num_classes):
    """Build a CNN model for Pokemon type multilabel classification"""
    # Use the functional API with Input layer as recommended
    inputs = layers.Input(shape=(*IMG_SIZE, 3))

    # First convolutional block
    x = layers.Conv2D(32, (3, 3), activation="relu")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)

    # Second convolutional block
    x = layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Third convolutional block
    x = layers.Conv2D(128, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Fourth convolutional block
    x = layers.Conv2D(128, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Flatten and dense layers
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)  # Dropout for regularization
    x = layers.Dense(512, activation="relu")(x)

    # For multilabel classification, use sigmoid activation (not softmax)
    outputs = layers.Dense(num_classes, activation="sigmoid")(x)

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
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training and Validation Accuracy")

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_history.png"))
    plt.close()

    # Also save the history data as CSV
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(output_dir, "training_history.csv"), index=False)


def get_predictions(model, X_test, datagen=None):
    """Get predictions for all test images"""
    all_probs = []  # Store all probabilities
    batch_size = BATCH_SIZE

    # Process test images in batches to avoid memory issues
    for i in range(0, len(X_test), batch_size):
        batch_paths = X_test[i : i + batch_size]
        batch_images = []

        for img_path in batch_paths:
            img_array = load_and_preprocess_image(img_path)
            batch_images.append(img_array)

        batch_images = np.array(batch_images, dtype=np.float32)
        batch_probs = model.predict(batch_images)  # Get probabilities
        all_probs.extend(batch_probs)  # Store all probabilities

    all_probs = np.array(all_probs)

    # For multilabel, we need to convert probabilities to binary predictions
    # using a threshold (typically 0.5)
    threshold = 0.5
    y_pred = (all_probs >= threshold).astype(int)

    return y_pred, all_probs


def generate_metrics(y_test, y_pred, label_encoder, output_dir):
    """Generate and save detailed metrics for multilabel classification"""
    # For multilabel, we need to compute metrics for each class
    class_names = label_encoder.classes_
    
    # Generate classification report for multilabel
    report = classification_report(
        y_test, y_pred, 
        target_names=class_names, 
        output_dict=True,
        zero_division=0
    )
    
    # Convert report to DataFrame for easier manipulation and saving
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(output_dir, "classification_report.csv"))
    
    # Print report to console
    print("\nClassification Report:")
    print(
        classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
    )

    # Generate per-class confusion matrices (for multilabel)
    mcm = multilabel_confusion_matrix(y_test, y_pred)

    # Create subplot for each type's confusion matrix
    fig, axes = plt.subplots(
        nrows=int(np.ceil(len(class_names) / 3)),
        ncols=3,
        figsize=(15, 5 * int(np.ceil(len(class_names) / 3))),
    )
    axes = axes.flatten()

    for i, (cm, class_name) in enumerate(zip(mcm, class_names)):
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Neg", "Pos"],
            yticklabels=["Neg", "Pos"],
            ax=axes[i],
        )
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("True")
        axes[i].set_title(f"Confusion Matrix - {class_name}")

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrices.png"))
    plt.close()

    # Calculate overall metrics averaged across classes
    accuracy = np.mean(np.all(y_test == y_pred, axis=1))
    print(f"\nExact Match Accuracy: {accuracy:.4f}")

    hamming_loss = np.mean(np.mean(y_test != y_pred, axis=1))
    print(f"Hamming Loss: {hamming_loss:.4f}")

    # Per-class metrics
    plt.figure(figsize=(12, 8))
    metrics_df = pd.DataFrame(
        {
            "Precision": [report[c]["precision"] for c in class_names],
            "Recall": [report[c]["recall"] for c in class_names],
            "F1-Score": [report[c]["f1-score"] for c in class_names],
        },
        index=class_names,
    )

    metrics_df.plot(kind="bar", figsize=(15, 8))
    plt.title("Per-Class Performance Metrics")
    plt.xlabel("Pokemon Type")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_class_metrics.png"))
    plt.close()


def display_predictions(model, X_test, y_test, label_encoder, output_dir):
    """Display predictions for a few test images with multilabel support"""
    # Select a few random test images
    num_samples = min(10, len(X_test))
    # Use a fixed random state for reproducible image selection
    viz_random_state = np.random.RandomState(seed=42)
    sample_indices = viz_random_state.choice(len(X_test), num_samples, replace=False)
    
    class_names = label_encoder.classes_
    
    plt.figure(figsize=(15, 10))
    for i, idx in enumerate(sample_indices):
        img_path = X_test[idx]
        true_labels = y_test[idx]

        try:
            # Load and preprocess the image
            img_array = load_and_preprocess_image(img_path)
            img_batch = np.expand_dims(img_array, axis=0)

            # Make prediction
            prediction = model.predict(img_batch)[0]
            predicted_labels = (prediction >= 0.5).astype(int)

            # Get true and predicted type names
            true_type_names = [
                class_names[j] for j in range(len(true_labels)) if true_labels[j] == 1
            ]
            pred_type_names = [
                class_names[j]
                for j in range(len(predicted_labels))
                if predicted_labels[j] == 1
            ]

            # Handle empty predictions
            if not pred_type_names:
                # Find the highest probability if no class exceeds threshold
                top_idx = np.argmax(prediction)
                pred_type_names = [
                    f"{class_names[top_idx]} ({prediction[top_idx]:.2f})"
                ]

            # Display the image with prediction
            plt.subplot(2, 5, i + 1)
            plt.imshow(img_array)
            plt.title(
                f"True: {', '.join(true_type_names)}\nPred: {', '.join(pred_type_names)}"
            )
            plt.axis("off")
        except Exception as e:
            print(f"Error displaying prediction for {img_path}: {str(e)}")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "test_predictions.png"))
    plt.close()


def save_test_predictions(model, X_test, y_test, label_encoder, output_dir):
    """Save predictions with logits for all test examples with multilabel support"""
    print("\nGenerating and saving prediction logits for all test examples...")

    # Get predictions and probabilities
    y_pred, all_probs = get_predictions(model, X_test)
    
    class_names = label_encoder.classes_

    # Create a DataFrame to store results
    results = []

    for i, (img_path, true_labels, pred_labels, probs) in enumerate(
        zip(X_test, y_test, y_pred, all_probs)
    ):
        # Get true and predicted type names
        true_type_names = [
            class_names[j] for j in range(len(true_labels)) if true_labels[j] == 1
        ]
        pred_type_names = [
            class_names[j] for j in range(len(pred_labels)) if pred_labels[j] == 1
        ]

        # Calculate correctness (exact match)
        correct = np.array_equal(true_labels, pred_labels)

        # Create a row with basic info
        row = {
            "example_id": i,
            "image_path": img_path,
            "true_types": ",".join(true_type_names),
            "predicted_types": ",".join(pred_type_names),
            "exact_match": correct,
        }

        # Add probabilities for each class
        for j, class_name in enumerate(class_names):
            row[f"prob_{class_name}"] = probs[j]

        results.append(row)

    # Convert to DataFrame and save
    results_df = pd.DataFrame(results)
    results_df.to_csv(
        os.path.join(output_dir, "test_predictions_with_logits.csv"), index=False
    )
    print(f"Saved predictions with logits for {len(results_df)} test examples")

    # Save top-3 predictions version
    save_top_predictions(results, all_probs, class_names, output_dir)

    return results_df


def save_top_predictions(results, all_probs, class_names, output_dir, top_n=3):
    """Save a version with just the top predictions for quick analysis"""
    top_results = []
    
    for i, row in enumerate(results):
        probs = all_probs[i]
        # Get top N predictions
        top_indices = np.argsort(probs)[::-1][:top_n]

        top_row = {
            "example_id": i,
            "image_path": row["image_path"],
            "true_types": row["true_types"],
            "exact_match": row["exact_match"],
        }

        # Add top N predictions with probabilities
        for rank, idx in enumerate(top_indices):
            class_name = class_names[idx]
            prob = probs[idx]
            top_row[f"pred{rank+1}"] = class_name
            top_row[f"prob{rank+1}"] = prob

        top_results.append(top_row)

    # Save top predictions
    top_df = pd.DataFrame(top_results)
    top_df.to_csv(os.path.join(output_dir, f"test_predictions_top{top_n}.csv"), index=False)
    print(f"Saved top {top_n} predictions for {len(top_df)} test examples")


def predict_single_image(image_path, show_image=True, top_k=5, output_dir=None):
    """Predict the types of a single Pokemon image with multilabel support"""
    # Create output directory if specified or needed
    if output_dir is None:
        output_dir = create_output_dir()

    # Find model and class files
    model_path = find_model_file(output_dir)
    class_file = find_class_file(output_dir)
    
    if model_path is None or class_file is None:
        return None, None

    # Load model and classes
    model = tf.keras.models.load_model(model_path)
    classes = np.load(class_file, allow_pickle=True)

    try:
        # Load and preprocess the image
        img_array = load_and_preprocess_image(image_path)
        img_batch = np.expand_dims(img_array, axis=0)

        # Get prediction probabilities
        prediction_probs = model.predict(img_batch)[0]

        # Sort indices by probability (highest first)
        sorted_indices = np.argsort(prediction_probs)[::-1]
        threshold = 0.5
        predicted_types = [
            classes[i] for i, prob in enumerate(prediction_probs) if prob >= threshold
        ]

        # If no types exceed threshold, take the highest one
        if not predicted_types and len(classes) > 0:
            top_idx = sorted_indices[0]
            predicted_types = [f"{classes[top_idx]} ({prediction_probs[top_idx]:.2f})"]

        # Display the image
        if show_image:
            visualize_prediction(img_array, predicted_types, classes, prediction_probs, sorted_indices, top_k, output_dir)

        # Print and save detailed predictions
        save_detailed_predictions(image_path, classes, prediction_probs, sorted_indices, threshold, output_dir)

        return classes, prediction_probs

    except Exception as e:
        print(f"Error predicting image {image_path}: {str(e)}")
        return None, None


def find_model_file(output_dir):
    """Find the model file to use for prediction"""
    model_paths = [
        "model.keras",
        "best_model.keras",
        "pokemon_type_model_1000.keras",
        "best_model_1000.keras",
    ]

    for path in model_paths:
        # Try both in current directory and in output dir
        if os.path.exists(path):
            print(f"Using model: {path}")
            return path
        elif os.path.exists(os.path.join(output_dir, path)):
            model_path = os.path.join(output_dir, path)
            print(f"Using model: {model_path}")
            return model_path
            
    print("Error: No model file found. Please train the model first.")
    return None


def find_class_file(output_dir):
    """Find the class file to use for prediction"""
    class_file_paths = ["valid_types.npy", "valid_types_1000.npy"]

    for path in class_file_paths:
        # Try both in current directory and in output dir
        if os.path.exists(path):
            print(f"Using classes from: {path}")
            return path
        elif os.path.exists(os.path.join(output_dir, path)):
            class_file = os.path.join(output_dir, path)
            print(f"Using classes from: {class_file}")
            return class_file
            
    print("Error: No class file found. Please train the model first.")
    return None


def visualize_prediction(img_array, predicted_types, classes, prediction_probs, sorted_indices, top_k, output_dir):
    """Create visualization of the prediction"""
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img_array)
    plt.title(f"Predicted Types:\n{', '.join(predicted_types)}")
    plt.axis("off")

    # Create a horizontal bar chart for probabilities
    plt.subplot(1, 2, 2)

    # Select top k predictions
    top_indices = sorted_indices[:top_k]
    top_probs = prediction_probs[top_indices]
    top_classes = classes[top_indices]

    # Horizontal bar chart
    y_pos = np.arange(len(top_indices))
    plt.barh(y_pos, top_probs, align="center")
    plt.yticks(y_pos, top_classes)
    plt.xlabel("Probability")
    plt.title("Top Predictions")

    plt.tight_layout()
    prediction_viz_path = os.path.join(output_dir, "prediction_probs.png")
    plt.savefig(prediction_viz_path)
    print(f"Visualization saved to {prediction_viz_path}")
    plt.show()


def save_detailed_predictions(image_path, classes, prediction_probs, sorted_indices, threshold, output_dir):
    """Save detailed prediction information to file"""
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
        confidence = "*" * stars
        print(f"{class_name:<15} {prob:.6f} {confidence}")

        pred_results.append(
            {
                "type": class_name,
                "probability": prob,
                "predicted": prob >= threshold,
            }
        )

    # Save to CSV
    pred_df = pd.DataFrame(pred_results)
    pred_csv_path = os.path.join(output_dir, "single_prediction_results.csv")
    pred_df.to_csv(pred_csv_path, index=False)
    print(f"Prediction results saved to {pred_csv_path}")


def predict_sample_folder(output_dir=None, sample_folder="sample"):
    """Predict types for all images in the sample folder (recursively)"""
    if output_dir is None:
        output_dir = create_output_dir()
    
    print(f"\nPredicting types for all images in the {sample_folder} folder (including subdirectories)...")
    
    # Find model and class files
    model_path = find_model_file(output_dir)
    class_file = find_class_file(output_dir)
    
    if model_path is None or class_file is None:
        print("Error: Model or class file not found.")
        return
    
    # Load model and classes
    model = tf.keras.models.load_model(model_path)
    classes = np.load(class_file, allow_pickle=True)
    
    # Find all image files in the sample folder recursively
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif']
    image_files = []
    
    # Use recursive globbing to find all images in subdirectories
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(sample_folder, f"**/*{ext}"), recursive=True))
        image_files.extend(glob.glob(os.path.join(sample_folder, f"**/*{ext.upper()}"), recursive=True))
    
    if not image_files:
        print(f"No image files found in {sample_folder} folder or its subdirectories.")
        return
    
    print(f"Found {len(image_files)} images in {sample_folder} folder and its subdirectories.")
    
    # Store prediction results
    results = []
    
    # Process each image
    for img_path in image_files:
        try:
            # Load and preprocess the image
            img_array = load_and_preprocess_image(img_path)
            img_batch = np.expand_dims(img_array, axis=0)
            
            # Get prediction probabilities
            prediction_probs = model.predict(img_batch)[0]
            
            # Sort indices by probability (highest first)
            sorted_indices = np.argsort(prediction_probs)[::-1]
            threshold = 0.5
            
            # Get predicted types
            predicted_types = [
                classes[i] for i, prob in enumerate(prediction_probs) if prob >= threshold
            ]
            
            # If no types exceed threshold, take the highest one
            if not predicted_types and len(classes) > 0:
                top_idx = sorted_indices[0]
                predicted_types = [classes[top_idx]]
            
            # Store detailed results for this image
            result = {
                "image_path": img_path,
                "filename": os.path.basename(img_path),
                "relative_path": os.path.relpath(img_path, sample_folder),
                "predicted_types": ",".join(predicted_types),
            }
            
            # Store top 3 predictions with probabilities
            for rank, idx in enumerate(sorted_indices[:3]):
                class_name = classes[idx]
                prob = prediction_probs[idx]
                result[f"pred{rank+1}"] = class_name
                result[f"prob{rank+1}"] = prob
                
            # Store all class probabilities
            for i, class_name in enumerate(classes):
                result[f"prob_{class_name}"] = prediction_probs[i]
                
            results.append(result)
            
            print(f"Processed: {os.path.relpath(img_path, sample_folder)} - Predicted: {', '.join(predicted_types)}")
            
        except Exception as e:
            print(f"Error processing image {img_path}: {str(e)}")
    
    # Save results to CSV
    if results:
        results_df = pd.DataFrame(results)
        results_csv_path = os.path.join(output_dir, "sample_predictions.csv")
        results_df.to_csv(results_csv_path, index=False)
        print(f"Sample prediction results saved to {results_csv_path}")
        
        # Create visualization of sample predictions
        visualize_sample_predictions(results_df, output_dir)
    
    return results


def visualize_sample_predictions(results_df, output_dir):
    """Create visualizations of sample predictions"""
    # Create a directory for sample images with predictions
    sample_viz_dir = os.path.join(output_dir, "sample_predictions")
    os.makedirs(sample_viz_dir, exist_ok=True)
    
    # Get top predicted type for each image
    results_df['top_type'] = results_df['pred1']
    
    # Count predictions by type
    type_counts = results_df['top_type'].value_counts()
    
    # Plot distribution of predicted types
    plt.figure(figsize=(12, 6))
    sns.barplot(x=type_counts.index, y=type_counts.values)
    plt.title("Distribution of Predicted Types in Sample Images")
    plt.xlabel("Pokémon Type")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sample_type_distribution.png"))
    plt.close()
    
    # Create a grid of sample images with predictions (up to 25 images)
    num_images = min(25, len(results_df))
    grid_size = int(np.ceil(np.sqrt(num_images)))
    
    plt.figure(figsize=(15, 15))
    for i in range(num_images):
        plt.subplot(grid_size, grid_size, i+1)
        
        img_path = results_df.iloc[i]['image_path']
        pred_types = results_df.iloc[i]['predicted_types']
        top_pred = results_df.iloc[i]['pred1']
        top_prob = results_df.iloc[i]['prob1']
        
        try:
            img = Image.open(img_path).convert("RGB")
            plt.imshow(img)
            plt.title(f"{top_pred}\n({top_prob:.2f})", fontsize=10)
            plt.axis("off")
        except Exception as e:
            print(f"Error displaying image {img_path}: {str(e)}")
            plt.text(0.5, 0.5, "Error loading image", ha='center', va='center')
            
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sample_predictions_grid.png"))
    plt.close()


def main():
    """Main function to train the Pokemon type classifier model and predict on samples"""
    print("Running Pokemon Type Classifier training...")
    output_dir = train_model()
    
    # After training, predict on sample folder if it exists
    sample_folder = "sample"
    if os.path.exists(sample_folder) and os.path.isdir(sample_folder):
        predict_sample_folder(output_dir, sample_folder)
    else:
        print(f"Sample folder '{sample_folder}' not found. Skipping sample predictions.")


if __name__ == "__main__":
    main()
