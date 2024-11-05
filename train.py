# train.py
import tensorflow as tf
from keras import layers, models, datasets
import matplotlib.pyplot as plt
from layers import SecondOrderPooling
from keras.callbacks import EarlyStopping

# Load the Fashion MNIST dataset
# (x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data() # mnist dataset code
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data() # cifar dataset code
# (x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data() # cifar dataset code

# Normalize and reshape the data
x_train = x_train.astype('float32') / 255.0
# x_train = x_train.reshape(-1, 28, 28, 1) # mnist dataset code
x_test = x_test.astype('float32') / 255.0
# x_test = x_test.reshape(-1, 28, 28, 1) # mnist dataset code

# Define the model using the Functional API
# inputs = layers.Input(shape=(28, 28, 1)) # mnist dataset code
inputs = layers.Input(shape=(32, 32, 3)) # cifar dataset code
x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
# x = SecondOrderPooling()(x) # global second order pooling
# x = layers.GlobalAveragePooling2D()(x) # global average pooling
x = layers.Flatten()(x) # no pooling control
x = layers.Dense(128, activation='relu')(x)
# outputs = layers.Dense(10, activation='softmax')(x) # mnist or cifar 10
outputs = layers.Dense(100, activation='softmax')(x) # cifar 100

model = models.Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model (TensorFlow will automatically use the GPU if available)
# model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test), callbacks=[early_stopping])

# Save the trained model in SavedModel format
model.save('second_order_pooling_model')  # Saves as a directory
