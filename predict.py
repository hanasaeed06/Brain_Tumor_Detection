import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_v3 import preprocess_input
import matplotlib.pyplot as plt

# Load trained model
model = load_model("models/tumor_detector.keras")

# Define class labels
class_labels = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]

# Function to preprocess image
def preprocess_image(image_path):
    img = cv2.imread(image_path)  # Read image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = cv2.resize(img, (150, 150))  # Resize to match training size
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array, img

# Function to predict tumor type
def predict_tumor(image_path):
    img_array, original_img = preprocess_image(image_path)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    predicted_label = class_labels[predicted_class]
    confidence = np.max(prediction) * 100  # Get confidence percentage
    return predicted_label, confidence, original_img

# Function to estimate tumor size using thresholding
def estimate_tumor_size(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read grayscale image
    img = cv2.resize(img, (150, 150))  # Resize to match training size
    _, thresholded = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)  # Thresholding
    tumor_size = np.sum(thresholded == 255)  # Count white pixels (tumor)
    return tumor_size, thresholded

# Function to highlight tumor in image
def highlight_tumor(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    _, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)  # Threshold to detect tumor
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find tumor boundary
    highlighted = img.copy()
    cv2.drawContours(highlighted, contours, -1, (255, 0, 0), 2)  # Draw contour in blue
    return highlighted

# Main execution
if __name__ == "__main__":
    image_path = input("Enter the path to the MRI image: ")  # Get user input for image path
    
    # Perform Predictions
    tumor_prediction, confidence, original_img = predict_tumor(image_path)
    tumor_size, thresholded_img = estimate_tumor_size(image_path)
    highlighted_img = highlight_tumor(image_path)

    # Display results
    print(f"Predicted Tumor Type: {tumor_prediction} ({confidence:.2f}% confidence)")
    print(f"Tumor Size: {tumor_size} pixels (approx.)")

    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(original_img)
    plt.title("Original MRI")
    
    plt.subplot(1, 3, 2)
    plt.imshow(thresholded_img, cmap="gray")
    plt.title("Tumor Segmentation")
    
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(highlighted_img, cv2.COLOR_BGR2RGB))
    plt.title("Tumor Highlighted")
    
    plt.show()
