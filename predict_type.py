import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

# Constants - must match those used in training
IMG_SIZE = (128, 128)

def load_model_and_classes():
    """Load the trained model and label encoder classes"""
    if not os.path.exists("pokemon_type_model.h5"):
        print("Error: Model file 'pokemon_type_model.h5' not found. Train the model first.")
        sys.exit(1)
        
    if not os.path.exists("label_encoder_classes.npy"):
        print("Error: Classes file 'label_encoder_classes.npy' not found. Train the model first.")
        sys.exit(1)
        
    # Load model
    model = tf.keras.models.load_model("pokemon_type_model.h5")
    
    # Load classes
    classes = np.load("label_encoder_classes.npy", allow_pickle=True)
    
    return model, classes

def predict_image(image_path, model, classes):
    """Predict the type of a Pokemon from an image"""
    try:
        # Load and preprocess the image
        img = Image.open(image_path).convert('RGB')
        img = img.resize(IMG_SIZE)
        img_array = np.array(img) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        prediction = model.predict(img_batch)[0]
        
        # Get top 3 predictions
        top_indices = prediction.argsort()[-3:][::-1]
        top_classes = classes[top_indices]
        top_probabilities = prediction[top_indices]
        
        # Print results
        print(f"\nPredictions for {os.path.basename(image_path)}:")
        for i, (class_name, prob) in enumerate(zip(top_classes, top_probabilities)):
            print(f"  {i+1}. {class_name}: {prob*100:.2f}%")
            
        # Return top prediction
        return top_classes[0], top_probabilities[0]
        
    except Exception as e:
        print(f"Error predicting image {image_path}: {e}")
        return None, 0

def display_prediction(image_path, predicted_type, probability):
    """Display the image with its prediction"""
    try:
        img = Image.open(image_path).convert('RGB')
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.title(f"Predicted: {predicted_type} ({probability*100:.2f}%)")
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"Error displaying image {image_path}: {e}")

# %%
# This file can be run as a Jupyter notebook with ipykernel
# Load model and classes
model, classes = load_model_and_classes()

# %%
# Get image path from command line or use a default
if len(sys.argv) > 1:
    image_path = sys.argv[1]
else:
    # You can set a default image path here
    image_path = input("Enter the path to a Pokémon image: ")

# Check if image exists
if not os.path.exists(image_path):
    print(f"Error: Image file '{image_path}' not found.")
else:
    # Predict image
    predicted_type, probability = predict_image(image_path, model, classes)
    
    if predicted_type is not None:
        # Display prediction
        display_prediction(image_path, predicted_type, probability)