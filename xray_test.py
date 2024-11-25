import os
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from layers import SecondOrderPooling
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths to the dataset and model
DATA_DIR = r"C:\Users\Total\.cache\kagglehub\datasets\nih-chest-xrays\data\versions\3"  # Replace with your actual path
TEST_CSV_PATH = os.path.join(DATA_DIR, "test_set.csv")
# MODEL_PATH = "second_order_pooling_model"  # Path to the saved model
MODEL_PATH = "global_average_pooling_model"
print(MODEL_PATH)
# Load the test dataset
print("Loading test dataset labels...")
test_df = pd.read_csv(TEST_CSV_PATH)

# Ensure 'Finding Labels' is simplified (in case it wasn't saved that way)
test_df['Finding Labels'] = test_df['Finding Labels'].apply(lambda x: x.split('|')[0].strip())

# Verify image paths
print("Verifying image paths...")
test_df = test_df[test_df['path'].apply(os.path.exists)]
print(f"Number of valid test image paths: {len(test_df)}")

# Image data generator for preprocessing
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

# Create a test data generator
test_gen = test_datagen.flow_from_dataframe(
    test_df,
    x_col='path',
    y_col='Finding Labels',
    target_size=(128, 128),  # Adjust based on your model input size
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Load the model
print("Loading the trained model...")
model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'SecondOrderPooling': SecondOrderPooling})

# Evaluate the model on the test set
print("Evaluating the model on the test set...")
test_loss, test_acc = model.evaluate(test_gen, verbose=2)
print(f'Test accuracy: {test_acc:.2f}')

# Make predictions on the test set
print("Making predictions...")
predictions = model.predict(test_gen)

# Visualize sample predictions
print("Displaying sample predictions...")
num_samples = 5
class_indices = {v: k for k, v in test_gen.class_indices.items()}  # Reverse class indices for label mapping

for i in range(num_samples):
    img_path = test_gen.filepaths[i]
    img = plt.imread(img_path)
    # Get the true label index and name
    true_label_index = np.argmax(test_gen[i][1][0])
    true_label = class_indices[true_label_index]
    # Get the predicted label index and name
    predicted_label_index = np.argmax(predictions[i])
    predicted_label = class_indices[predicted_label_index]

    plt.figure(figsize=(4, 4))
    if img.ndim == 2 or img.shape[-1] == 1:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    plt.title(f"True Label: {true_label}, Predicted: {predicted_label}")
    plt.axis('off')
    plt.show()
