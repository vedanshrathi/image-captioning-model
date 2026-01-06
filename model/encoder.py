import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50

def build_encoder():
    base = ResNet50(weights="imagenet", include_top=False)
    base.trainable = False
    return base
