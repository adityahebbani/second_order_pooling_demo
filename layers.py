# layers.py (Create this file in the same directory as your training and testing scripts)
import tensorflow as tf
from tensorflow.keras import layers

class SecondOrderPooling(layers.Layer):
    def __init__(self, **kwargs):
        super(SecondOrderPooling, self).__init__(**kwargs)
        self.output_dim = None
        self.channels = None

    def build(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError('SecondOrderPooling expects inputs with shape (batch_size, height, width, channels)')
        if input_shape[-1] is None:
            raise ValueError('The channel dimension of the inputs should be defined. Found None.')
        self.channels = int(input_shape[-1])
        self.output_dim = self.channels * self.channels
        super(SecondOrderPooling, self).build(input_shape)

    def call(self, inputs):
        # inputs: (batch_size, height, width, channels)
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        channels = self.channels

        # Reshape inputs to (batch_size, height*width, channels)
        x = tf.reshape(inputs, [batch_size, height * width, channels])

        # Center the data
        x_mean = tf.reduce_mean(x, axis=1, keepdims=True)
        x_centered = x - x_mean

        # Compute covariance matrix: (batch_size, channels, channels)
        cov = tf.matmul(x_centered, x_centered, transpose_a=True) / tf.cast(height * width, tf.float32)

        # Flatten the covariance matrices
        cov_flat = tf.reshape(cov, [batch_size, self.output_dim])

        return cov_flat

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        base_config = super(SecondOrderPooling, self).get_config()
        return base_config
