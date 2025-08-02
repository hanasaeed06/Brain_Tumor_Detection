from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
import cv2
import tensorflow as tf
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = "static/uploads/"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load trained model
model = load_model("models/tumor_detector.keras")
class_labels = ["Glioma", "Meningioma", "no Tumor", "Pituitary"]

# Function to check allowed file types
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to preprocess image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, img

# Function to predict tumor type
def predict_tumor(image_path):
    img_array, original_img = preprocess_image(image_path)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    predicted_label = class_labels[predicted_class]
    confidence = np.max(prediction) * 100
    return predicted_label, confidence, original_img

# Function to highlight tumor
def highlight_tumor(image_path, prediction):
    if prediction == "no Tumor":
        return None  # No highlighting if there's no tumor
    
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:  # If no tumor is detected in contours, return None
        return None

    highlighted = img.copy()
    cv2.drawContours(highlighted, contours, -1, (255, 0, 0), 2)

    highlighted_path = os.path.join(app.config["UPLOAD_FOLDER"], "highlighted_" + os.path.basename(image_path))
    cv2.imwrite(highlighted_path, highlighted)

    return "highlighted_" + os.path.basename(image_path)  # Return the filename, not the path


# Flask Routes
# Flask Routes
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        
        file = request.files["file"]
        if file.filename == "" or not allowed_file(file.filename):
            return redirect(request.url)
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Get predictions
        tumor_prediction, confidence, original_img = predict_tumor(filepath)
        
        # Get highlighted image only if tumor is present
        highlighted_img = highlight_tumor(filepath, tumor_prediction)

        return render_template("result.html", filename=filename, 
                               prediction=tumor_prediction, confidence=confidence, 
                               highlighted_img=highlighted_img)
    return render_template("index.html")



@app.route("/display/<filename>")
def display_image(filename):
    return redirect(url_for("static", filename="uploads/" + filename))

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)