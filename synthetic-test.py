# test_sop_vs_gap.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from layers import SecondOrderPooling  # Importing from layers.py

# Parameters
num_classes = 3
image_size = 16  # Must match training image size
class_names = ['Identity', 'Block', 'Stripe']

# Function to generate a single covariance matrix per class
def generate_covariance_matrix(image_size, covariance_type):
    if covariance_type == 'identity':
        cov = np.identity(image_size * image_size)
    elif covariance_type == 'block':
        cov = np.zeros((image_size * image_size, image_size * image_size))
        block_size = image_size // 4
        for i in range(0, image_size * image_size, block_size * image_size):
            idx = slice(i, i + block_size * image_size)
            cov[idx, idx] = 1
    elif covariance_type == 'stripe':
        cov = np.zeros((image_size * image_size, image_size * image_size))
        for i in range(image_size):
            idx = np.arange(i * image_size, (i + 1) * image_size)
            cov[np.ix_(idx, idx)] = 1
    else:
        raise ValueError("Invalid covariance_type")
    # Add a small value to the diagonal for numerical stability
    cov += np.eye(image_size * image_size) * 1e-5
    return cov

# Function to generate all samples for a class using a precomputed covariance matrix
def generate_gaussian_images(num_samples, image_size, covariance_matrix):
    mean = np.zeros(image_size * image_size)
    # Efficiently generate multiple samples at once
    samples = np.random.multivariate_normal(mean, covariance_matrix, num_samples)
    samples = samples.reshape(num_samples, image_size, image_size, 1)
    # Normalize to have zero mean and unit variance
    samples = (samples - np.mean(samples, axis=(1,2,3), keepdims=True)) / (np.std(samples, axis=(1,2,3), keepdims=True) + 1e-5)
    return samples

print("Generating synthetic test dataset...")
# Define covariance types for each class
covariance_types = ['identity', 'block', 'stripe']

# Generate images and labels for each class
images = []
labels = []
for class_label in range(num_classes):
    cov_type = covariance_types[class_label]
    cov_matrix = generate_covariance_matrix(image_size, cov_type)
    class_images = generate_gaussian_images(500, image_size, cov_matrix)  # 500 samples per class
    images.append(class_images)
    labels.append(np.full(500, class_label))

# Combine all classes
X_test = np.concatenate(images, axis=0)
y_test = np.concatenate(labels, axis=0)

# Shuffle dataset
indices = np.arange(len(X_test))
np.random.shuffle(indices)
X_test = X_test[indices]
y_test = y_test[indices]

# Load models
print("Loading models...")

# Load SOP model with custom layer
model_sop = load_model('model_sop', custom_objects={'SecondOrderPooling': SecondOrderPooling})

# Load GAP model
model_gap = load_model('model_gap')

# Convert labels to categorical
y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)

# Evaluate models
print("\nEvaluating model with Second Order Pooling...")
loss_sop, acc_sop = model_sop.evaluate(X_test, y_test_cat, verbose=0)
print(f"Test Accuracy with Second Order Pooling: {acc_sop:.4f}")

print("\nEvaluating model with Global Average Pooling...")
loss_gap, acc_gap = model_gap.evaluate(X_test, y_test_cat, verbose=0)
print(f"Test Accuracy with Global Average Pooling: {acc_gap:.4f}")

# Predictions
y_pred_sop = np.argmax(model_sop.predict(X_test), axis=1)
y_pred_gap = np.argmax(model_gap.predict(X_test), axis=1)

# Classification Reports
print("\nClassification Report for Second Order Pooling Model:")
print(classification_report(y_test, y_pred_sop, target_names=class_names))

print("\nClassification Report for Global Average Pooling Model:")
print(classification_report(y_test, y_pred_gap, target_names=class_names))

# Confusion Matrices
print("\nConfusion Matrix for Second Order Pooling Model:")
print(confusion_matrix(y_test, y_pred_sop))

print("\nConfusion Matrix for Global Average Pooling Model:")
print(confusion_matrix(y_test, y_pred_gap))

# Visualization (Optional)
def visualize_predictions(X, y_true, y_pred, title):
    indices = np.random.choice(len(X), size=5, replace=False)
    plt.figure(figsize=(15, 3))
    for i, idx in enumerate(indices):
        plt.subplot(1, 5, i+1)
        plt.imshow(X[idx].squeeze(), cmap='gray')
        plt.title(f"True: {class_names[int(y_true[idx])]} \nPred: {class_names[int(y_pred[idx])]}")
        plt.axis('off')
    plt.suptitle(title)
    plt.show()

print("\nSample Predictions for Second Order Pooling Model:")
visualize_predictions(X_test, y_test, y_pred_sop, "Second Order Pooling Predictions")

print("\nSample Predictions for Global Average Pooling Model:")
visualize_predictions(X_test, y_test, y_pred_gap, "Global Average Pooling Predictions")
