# synthetic-train.py

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import os
from layers import SecondOrderPooling  # Importing from layers.py

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Parameters
num_classes = 3
images_per_class = 500  # Reduced from 1000 for computational efficiency
image_size = 16          # Reduced from 32 for faster processing
batch_size = 64
epochs = 30

# Generate synthetic dataset with identical first-order statistics
def generate_gaussian_images(num_samples, image_size, covariance_type):
    """
    Generate images with zero mean and unit variance, differing only in covariance structure.
    covariance_type: 'identity', 'block', 'stripe'
    """
    images = []
    for _ in range(num_samples):
        mean = np.zeros(image_size * image_size)
        if covariance_type == 'identity':
            cov = np.identity(image_size * image_size)
        elif covariance_type == 'block':
            # Block covariance
            cov = np.zeros((image_size * image_size, image_size * image_size))
            block_size = image_size // 4
            for i in range(0, image_size * image_size, block_size * image_size):
                idx = slice(i, i + block_size * image_size)
                cov[idx, idx] = 1
        elif covariance_type == 'stripe':
            # Stripe covariance
            cov = np.zeros((image_size * image_size, image_size * image_size))
            for i in range(image_size):
                idx = np.arange(i * image_size, (i + 1) * image_size)
                cov[np.ix_(idx, idx)] = 1
        else:
            raise ValueError("Invalid covariance_type")

        # Add a small value to the diagonal for numerical stability
        cov += np.eye(image_size * image_size) * 1e-5

        # Generate a random sample from the multivariate normal distribution
        img = np.random.multivariate_normal(mean, cov)
        img = img.reshape(image_size, image_size)
        # Normalize to have zero mean and unit variance
        img = (img - np.mean(img)) / (np.std(img) + 1e-5)
        images.append(img)
    return np.array(images)

print("Generating synthetic dataset...")
# Class 0: Identity covariance
images_class0 = generate_gaussian_images(images_per_class, image_size, 'identity')
labels_class0 = np.zeros(images_per_class)

# Class 1: Block covariance
images_class1 = generate_gaussian_images(images_per_class, image_size, 'block')
labels_class1 = np.ones(images_per_class)

# Class 2: Stripe covariance
images_class2 = generate_gaussian_images(images_per_class, image_size, 'stripe')
labels_class2 = np.full(images_per_class, 2)

# Combine and create labels
images = np.concatenate((images_class0, images_class1, images_class2), axis=0)
labels = np.concatenate((labels_class0, labels_class1, labels_class2), axis=0)

# Expand dimensions to add channel dimension
images = np.expand_dims(images, axis=-1)  # Shape: (samples, height, width, channels)

# Shuffle dataset
indices = np.arange(len(images))
np.random.shuffle(indices)
images = images[indices]
labels = labels[indices]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, stratify=labels, random_state=42
)

# Convert labels to categorical
y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)

# Define models
def create_model_sop(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = SecondOrderPooling()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs, outputs)

def create_model_gap(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs, outputs)

# Compile models
input_shape = (image_size, image_size, 1)

model_sop = create_model_sop(input_shape, num_classes)
model_gap = create_model_gap(input_shape, num_classes)

optimizer = optimizers.Adam(learning_rate=0.001)

model_sop.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model_gap.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

# Train models
print("\nTraining model with Second Order Pooling...")
history_sop = model_sop.fit(
    X_train, y_train_cat,
    epochs=epochs,
    validation_data=(X_test, y_test_cat),
    callbacks=[early_stopping],
    batch_size=batch_size,
    verbose=1
)

print("\nTraining model with Global Average Pooling...")
history_gap = model_gap.fit(
    X_train, y_train_cat,
    epochs=epochs,
    validation_data=(X_test, y_test_cat),
    callbacks=[early_stopping],
    batch_size=batch_size,
    verbose=1
)

# Save models
model_sop.save('model_sop')
model_gap.save('model_gap')
print("\nModels saved successfully.")
