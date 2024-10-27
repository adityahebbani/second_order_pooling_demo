import tensorflow as tf

# Define the custom second-order pooling layer
class SecondOrderPooling(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SecondOrderPooling, self).__init__(**kwargs)

    def call(self, inputs):
        batch_size, height, width, channels = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2], tf.shape(inputs)[3]
        
        # Reshape input to batch_size x (height * width) x channels
        reshaped = tf.reshape(inputs, [batch_size, height * width, channels])
        
        # Compute the covariance matrix (second-order pooling)
        cov_matrix = tf.matmul(reshaped, reshaped, transpose_a=True) / tf.cast(height * width, tf.float32)
        
        # Flatten the covariance matrix for the next layer
        flattened_cov_matrix = tf.reshape(cov_matrix, [batch_size, -1])

        # print covariance matrix
        print("SecondOrderPooling Output:")
        print(flattened_cov_matrix)
        
        return flattened_cov_matrix

    def compute_output_shape(self, input_shape):
        batch_size, height, width, channels = input_shape
        return (batch_size, channels * channels)