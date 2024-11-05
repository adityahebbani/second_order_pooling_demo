# layers.py
import tensorflow as tf

# Define the custom second-order pooling layer
class SecondOrderPooling(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SecondOrderPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError('SecondOrderPooling expects inputs with shape (batch_size, height, width, channels)')
        self.channels = int(input_shape[-1])
        super(SecondOrderPooling, self).build(input_shape)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        channels = self.channels  # Static number of channels

        # Reshape input to (batch_size, height * width, channels)
        reshaped = tf.reshape(inputs, [batch_size, height * width, channels])

        # Compute the covariance matrix (second-order pooling)
        cov_matrix = tf.matmul(reshaped, reshaped, transpose_a=True) / tf.cast(height * width, tf.float32)

        # Flatten the covariance matrix for the next layer
        flattened_cov_matrix = tf.reshape(cov_matrix, [batch_size, channels * channels])

        return flattened_cov_matrix

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        channels = input_shape[-1]
        return (batch_size, channels * channels)

    def get_config(self):
        base_config = super(SecondOrderPooling, self).get_config()
        return base_config
