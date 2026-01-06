import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences

# MUST match training
MAX_LEN = 38


# --------------------------------------------------
# Extract CNN features from image
# --------------------------------------------------
def extract_features(image, cnn):
    image = image.resize((224, 224))
    image = np.array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)

    features = cnn.predict(image, verbose=0)
    features = tf.reshape(
        features, (features.shape[0], -1, features.shape[3])
    )
    return features   # (1, 49, 2048)


# --------------------------------------------------
# Temperature-based sampling (prevents repetition)
# --------------------------------------------------
def sample_with_temperature(preds, temperature=0.8):
    preds = np.asarray(preds).astype("float64")

    # Avoid log(0)
    preds = np.log(preds + 1e-8) / temperature

    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)

    return np.random.choice(len(preds), p=preds)


# --------------------------------------------------
# Generate caption
# --------------------------------------------------
def generate_caption(model, tokenizer, cnn, image):
    caption = "<start>"
    last_word = None

    for _ in range(MAX_LEN):
        # Convert caption to sequence
        seq = tokenizer.texts_to_sequences([caption])[0]
        seq = pad_sequences([seq], MAX_LEN, padding="post")

        # Extract image features
        image_features = extract_features(image, cnn)

        # Predict next word
        preds = model.predict(
            (image_features, seq),
            verbose=0
        )[0]

        next_word_id = sample_with_temperature(preds, temperature=0.8)
        word = tokenizer.index_word.get(next_word_id)

        # Stop conditions
        if word is None:
            break

        if word == "<end>":
            break

        # Prevent repeating same word
        if word == last_word:
            break

        # Skip very short meaningless words
        if len(word) <= 1:
            continue

        caption += " " + word
        last_word = word

    # Clean output
    caption = caption.replace("<start>", "").strip()
    return caption
