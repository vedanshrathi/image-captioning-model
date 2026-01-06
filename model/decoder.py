import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, units):
        super().__init__()
        self.embed = Embedding(vocab_size, embedding_dim)
        self.lstm = LSTM(units, return_sequences=True, return_state=True)
        self.fc = Dense(vocab_size)

    def call(self, x):
        x = self.embed(x)
        x, _, _ = self.lstm(x)
        return self.fc(x)
