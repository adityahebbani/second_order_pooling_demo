import tensorflow as tf
import keras
from keras import datasets
import matplotlib.pyplot as plt
import numpy as np
from layers import SecondOrderPooling

# Load the Fashion MNIST dataset
(_, _), (x_test, y_test) = datasets.fashion_mnist.load_data()

# Normalize and reshape test data
x_test = x_test.astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1)

# Load the model from the SavedModel format
model = tf.keras.models.load_model('second_order_pooling_model.keras', custom_objects={'SecondOrderPooling': SecondOrderPooling})

# Make predictions on the test set
predictions = model.predict(x_test)

# Show some sample predictions
num_samples = 5
for i in range(num_samples):
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"True Label: {y_test[i]}, Predicted: {np.argmax(predictions[i])}")
    plt.show()

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'Test accuracy: {test_acc}')