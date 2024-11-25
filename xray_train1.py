import os
import pandas as pd
import numpy as np
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping
from layers import SecondOrderPooling

# Define paths
DATA_DIR = r"C:\Users\Total\.cache\kagglehub\datasets\nih-chest-xrays\data\versions\3"
CSV_PATH = os.path.join(DATA_DIR, "Data_Entry_2017.csv")

# Load Dataset and Labels
print("Loading dataset labels...")
df = pd.read_csv(CSV_PATH)
print("First few rows of the dataset:")
print(df.head())

# Add Full Paths for Image Files
def get_image_path(filename):
    # Iterate through image folders to find the file
    for i in range(1, 13):  # 12 subdirectories named images_001 to images_012
        folder_name = f"images_{i:03}\\images"  # images_001, images_002, ...
        folder_path = os.path.join(DATA_DIR, folder_name)
        full_path = os.path.join(folder_path, filename)
        if os.path.exists(full_path):
            return full_path
    return None

print("Updating image paths...")
df['path'] = df['Image Index'].apply(get_image_path)
print("Updated DataFrame after adding image paths:")
print(df.head())  # Check if 'Finding Labels' and 'path' columns exist
print("Columns:", df.columns)

# Drop rows where the image file was not found
df = df.dropna(subset=['path'])
print(f"Number of valid image paths: {len(df)}")

# Ensure there are valid image paths
if len(df) == 0:
    print("No valid image paths found. Please check your DATA_DIR and extraction steps.")
    exit(1)

# Simplify Labels
df['Finding Labels'] = df['Finding Labels'].apply(lambda x: x.split('|')[0].strip())

# Remove any rows where 'Finding Labels' is empty
df = df[df['Finding Labels'] != '']

# Ensure consistent data splitting by setting a fixed random seed
random_seed = 42

# Split Dataset into Training, Validation, and Test Sets
print("Splitting dataset into training, validation, and test sets...")

# First, shuffle the data
df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

# Calculate split indices
train_frac = 0.8
val_frac = 0.1  # 10% for validation
test_frac = 0.1  # 10% for testing

train_end = int(len(df) * train_frac)
val_end = train_end + int(len(df) * val_frac)

train_df = df[:train_end]
val_df = df[train_end:val_end]
test_df = df[val_end:]

# Save the test set to a CSV file for use in the test script
TEST_CSV_PATH = os.path.join(DATA_DIR, "test_set.csv")
test_df.to_csv(TEST_CSV_PATH, index=False)
print(f"Test set saved to {TEST_CSV_PATH}")

# Calculate Class Weights
print("Calculating class weights...")
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_df['Finding Labels']),
    y=train_df['Finding Labels']
)
class_weights_dict = dict(zip(np.unique(train_df['Finding Labels']), class_weights))
print("Class weights:", class_weights_dict)

# Map class weights to class indices
label_to_index = {label: index for index, label in enumerate(train_df['Finding Labels'].unique())}
class_weights_indexed = {label_to_index[label]: weight for label, weight in class_weights_dict.items()}

# Define Data Generators
print("Creating data generators...")

# Training Data Generator with Augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    horizontal_flip=True,
    rotation_range=10
)

# Validation Data Generator without Augmentation
val_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0
)

train_gen = train_datagen.flow_from_dataframe(
    train_df,
    x_col='path',
    y_col='Finding Labels',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

val_gen = val_datagen.flow_from_dataframe(
    val_df,
    x_col='path',
    y_col='Finding Labels',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

# Define the CNN Model
print("Defining the model...")
inputs = layers.Input(shape=(128, 128, 3))
x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
# x = SecondOrderPooling()(x)  # Using global second order pooling
x = layers.GlobalAveragePooling2D()(x) # Using global average pooling
x = layers.Dense(128, activation='relu')(x)
outputs = layers.Dense(len(train_gen.class_indices), activation='softmax')(x)

model = models.Model(inputs=inputs, outputs=outputs)

# Compile the Model with a Smaller Learning Rate
print("Compiling the model...")
optimizer = optimizers.Adam(learning_rate=0.0001)  # Adjusted learning rate
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the Model
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
print("Starting training...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,  # Adjust epochs as needed
    callbacks=[early_stopping],
    class_weight=class_weights_indexed,
    verbose=1
)

# Save the Trained Model
print("Saving the model...")
model.save('global_average_pooling_model')  # Saves as a directory
print("Training complete. Model saved as 'second_order_pooling_model'.")
