from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import pickle
from PIL import Image

from inference import generate_caption

app = Flask(__name__)

# Load trained caption model
model = tf.keras.models.load_model("saved_model/caption_model.h5")

# Load tokenizer
with open("saved_model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load CNN (same as training)
cnn = tf.keras.applications.ResNet50(
    weights="imagenet",
    include_top=False
)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"caption": "No image uploaded"})

    image_file = request.files["image"]
    image = Image.open(image_file).convert("RGB")

    caption = generate_caption(model, tokenizer, cnn, image)

    return jsonify({"caption": caption})

if __name__ == "__main__":
    app.run(debug=True)
