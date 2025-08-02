import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os

# Define dataset path
train_dir = "dataset/train"
val_dir = "dataset/test"

# Define image size & batch size
IMAGE_SIZE = (150, 150)  # Keep consistent with predict.py
BATCH_SIZE = 32
EPOCHS = 30  # Increased for better training

# Data Augmentation to improve generalization
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Load dataset
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# Calculate class weights for balanced training
categories = ['glioma', 'meningioma', 'pituitary', 'notumor']
class_counts = [len(os.listdir(os.path.join(train_dir, cat))) for cat in categories]
class_weights = compute_class_weight('balanced', classes=np.arange(len(categories)), 
                                     y=np.concatenate([[i] * c for i, c in enumerate(class_counts)]))
class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)

# Load InceptionV3 model (pretrained on ImageNet)
base_model = InceptionV3(weights="imagenet", include_top=False, input_shape=(150, 150, 3))
base_model.trainable = False  # Freeze base model

# Add custom layers
x = Flatten()(base_model.output)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(4, activation="softmax")(x)  # 4 categories

# Compile model
model = Model(inputs=base_model.input, outputs=x)
model.compile(optimizer=Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])

# Save best model using checkpointing
checkpoint = tf.keras.callbacks.ModelCheckpoint("models/best_model.keras", monitor="val_accuracy", save_best_only=True)

# Train model with class weights
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[checkpoint]
)

# Save final model
model.save("models/tumor_detector.keras")
