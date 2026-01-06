# 🖼️ Image Caption Generation System (CNN + LSTM + Attention)

This project implements an **Image Caption Generation System** using **Deep Learning**, where a model automatically generates a natural language description for a given image.

It uses:
- **ResNet50 (CNN)** for image feature extraction
- **LSTM** for sequence modeling
- **Attention Mechanism** to focus on relevant image regions
- **Flask + HTML/CSS/JS** for a simple web interface

---

## 📌 Features

- Upload an image and generate a caption
- Attention-based image captioning model
- Web-based frontend using Flask
- Trained using real-world datasets (Flickr8k / MS COCO)
- Suitable for **Final Year AI / ML Project**

---

## 🧠 Model Architecture

```
Image → ResNet50 → Feature Maps
                     ↓
                Attention Layer
                     ↓
Text Input → Embedding → LSTM → Dense → Caption
```

---

## 📁 Project Structure

```
image_captioning/
│
├── app.py                    # Flask backend
├── inference.py              # Caption generation logic
├── model/
│   └── train.py              # Model training (with attention)
│
├── dataset/
│   ├── Images/               # Image dataset
│   └── captions.txt          # Image-caption mapping
│
├── saved_model/
│   ├── caption_model.h5      # Trained model (auto-generated)
│   └── tokenizer.pkl         # Tokenizer (auto-generated)
│
├── templates/
│   └── index.html            # Frontend UI
│
├── static/
│   ├── style.css             # Styling
│   └── script.js             # Frontend logic
│
├── venv/                     # Virtual environment
└── README.md                 # Project documentation
```

---

## 🧪 Dataset Used

Recommended datasets:
- **MS COCO (Best accuracy)**
- Flickr30k
- Flickr8k (for learning/demo)

Each image has multiple captions to improve learning quality.

---

## ⚙️ Installation & Setup

### 1️⃣ Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 2️⃣ Install Dependencies
```bash
pip install tensorflow flask numpy pillow nltk matplotlib tqdm
```

---

## 🏋️ Train the Model

⚠️ Do NOT create model files manually.

```bash
python model/train.py
```

This will automatically generate:
- `saved_model/caption_model.h5`
- `saved_model/tokenizer.pkl`

---

## 🚀 Run the Web Application

```bash
python app.py
```

Open browser:
```
http://127.0.0.1:5000
```

Upload an image and click **Generate Caption**.

---

## 📸 Sample Output

**Input Image:**  
A person riding a bike on the road

**Generated Caption:**  
> *A man riding a bike on the street*

---

## 🎓 Viva Explanation (Short)

> This project uses a CNN–LSTM architecture with an attention mechanism to generate natural language captions for images. The CNN extracts visual features, attention focuses on important regions, and the LSTM generates captions word by word.

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- OpenCV (optional)
- Flask
- HTML, CSS, JavaScript

---

## 📈 Future Enhancements

- Beam Search decoding
- Attention heatmap visualization
- Voice-based caption output
- VQA (Visual Question Answering)

---

## 👨‍🎓 Author

Final Year AI / ML Project  
Developed for academic and learning purposes.

---

⭐ *If you like this project, feel free to extend or improve it!* ⭐
