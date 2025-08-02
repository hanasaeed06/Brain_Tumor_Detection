import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, image
from tensorflow.keras.models import load_model

# Image settings
IMG_SIZE = 150  # Image size for training and prediction
BATCH_SIZE = 32
CONFIDENCE_THRESHOLD = 0.7  # Set a confidence threshold for predictions

def load_data(data_dir):
    """ Load dataset and apply augmentation """
    datagen = ImageDataGenerator(
        rescale=1./255, 
        validation_split=0.2,
        rotation_range=20,  # Data augmentation
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    train_data = datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    val_data = datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    return train_data, val_data

def preprocess_uploaded_image(img_path):
    """ Load and preprocess an uploaded image for model prediction """
    img = cv2.imread(img_path)  # Read image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Ensure RGB format
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    print("Preprocessed Image Shape:", img.shape)  # Debugging
    return img

def predict_image(model, img_path, class_names):
    """ Predict the class of an uploaded image """
    img_array = preprocess_uploaded_image(img_path)
    predictions = model.predict(img_array)
    
    confidence = np.max(predictions)
    predicted_class = np.argmax(predictions)

    if confidence > CONFIDENCE_THRESHOLD:
        print(f"Predicted Class: {class_names[predicted_class]} with confidence {confidence:.2f}")
    else:
        print(f"Low confidence prediction: {class_names[predicted_class]} with confidence {confidence:.2f}")

# Example usage
if __name__ == "__main__":
    model_path = "your_model.h5"  # Update with your model path
    image_path = "your_uploaded_image.jpg"  # Update with the test image path
    class_labels = ['class_1', 'class_2', 'class_3']  # Update based on your dataset

    model = load_model(model_path)  # Load the trained model
    predict_image(model, image_path, class_labels)  # Predict an uploaded image
