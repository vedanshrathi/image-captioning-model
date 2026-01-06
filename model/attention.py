import tensorflow as tf
from tensorflow.keras.layers import Dense

class Attention(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, features, hidden):
        hidden = tf.expand_dims(hidden, 1)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden))
        weights = tf.nn.softmax(self.V(score), axis=1)
        context = weights * features
        return tf.reduce_sum(context, axis=1)
