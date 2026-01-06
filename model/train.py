import os
import pickle
import numpy as np
import string
from PIL import Image

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import (
    Input, Dense, Embedding, LSTM,
    Add, Multiply, GlobalAveragePooling1D
)
from tensorflow.keras.models import Model

# =====================================================
# PATHS
# =====================================================
IMAGE_DIR = "dataset/Images"
CAPTIONS_FILE = "dataset/captions.txt"
SAVE_DIR = "saved_model"

os.makedirs(SAVE_DIR, exist_ok=True)

# =====================================================
# LOAD & CLEAN CAPTIONS (HEADER SAFE)
# =====================================================
def load_captions(filename):
    mapping = {}
    with open(filename, "r", encoding="utf-8") as f:
        next(f)  # skip header: image,caption
        for line in f:
            tokens = line.strip().split(",", 1)
            if len(tokens) < 2:
                continue

            image_id = tokens[0].split(".")[0]
            caption = tokens[1].lower()
            caption = caption.translate(
                str.maketrans("", "", string.punctuation)
            )
            caption = "<start> " + caption + " <end>"
            mapping.setdefault(image_id, []).append(caption)
    return mapping

captions = load_captions(CAPTIONS_FILE)
print("Total images:", len(captions))

# =====================================================
# TOKENIZATION
# =====================================================
all_captions = []
for caps in captions.values():
    all_captions.extend(caps)

tokenizer = Tokenizer(oov_token="<unk>")
tokenizer.fit_on_texts(all_captions)

vocab_size = len(tokenizer.word_index) + 1
max_len = max(len(c.split()) for c in all_captions)

print("Vocabulary size:", vocab_size)
print("Max caption length:", max_len)

# Save tokenizer
with open(os.path.join(SAVE_DIR, "tokenizer.pkl"), "wb") as f:
    pickle.dump(tokenizer, f)

# =====================================================
# CNN FEATURE EXTRACTOR
# =====================================================
cnn = ResNet50(weights="imagenet", include_top=False)
cnn.trainable = False

def extract_features(image_path):
    img = Image.open(image_path).resize((224, 224))
    img = np.array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    features = cnn.predict(img, verbose=0)
    features = features.reshape(
        (features.shape[0], -1, features.shape[3])
    )
    return features  # (1, 49, 2048)

# =====================================================
# DATA GENERATOR
# =====================================================
def data_generator(captions):
    while True:
        for img_id, caps in captions.items():
            img_path = os.path.join(IMAGE_DIR, img_id + ".jpg")

            if not os.path.exists(img_path):
                continue

            image_features = extract_features(img_path)

            for cap in caps:
                seq = tokenizer.texts_to_sequences([cap])[0]
                for i in range(1, len(seq)):
                    in_seq = pad_sequences(
                        [seq[:i]],
                        max_len,
                        padding="post"
                    )
                    out_seq = np.array([seq[i]])
                    yield ((image_features, in_seq), out_seq)

# =====================================================
# ATTENTION MODEL (KERAS SAFE)
# =====================================================
image_input = Input(shape=(49, 2048))
caption_input = Input(shape=(max_len,))

# Image projection
image_dense = Dense(256, activation="relu")(image_input)

# Caption embedding
caption_embed = Embedding(vocab_size, 256)(caption_input)

# LSTM
lstm_out, _, _ = LSTM(
    256,
    return_state=True
)(caption_embed)

# Attention mechanism
attn_scores = Dense(256, activation="tanh")(image_dense)
attn_scores = Dense(1, activation="softmax")(attn_scores)

context = Multiply()([image_dense, attn_scores])
context = GlobalAveragePooling1D()(context)

# Combine context + LSTM output
decoder = Add()([context, lstm_out])

# Output layer
outputs = Dense(vocab_size, activation="softmax")(decoder)

model = Model(
    inputs=[image_input, caption_input],
    outputs=outputs
)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy"
)

model.summary()

# =====================================================
# TRAIN MODEL
# =====================================================
model.fit(
    data_generator(captions),
    steps_per_epoch=len(captions),
    epochs=25
)

# =====================================================
# SAVE MODEL
# =====================================================
model.save(os.path.join(SAVE_DIR, "caption_model.h5"))
print("✅ Attention model trained and saved successfully.")
