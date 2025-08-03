
import argparse
import numpy as np
import tensorflow as tf
import json
from PIL import Image
import os

def process_image(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    image = np.asarray(image) / 255.0
    return image

def predict(image_path, model, top_k):
    image = process_image(image_path)
    image = np.expand_dims(image, axis=0)
    predictions = model.predict(image)[0]
    top_indices = predictions.argsort()[-top_k:][::-1]
    top_probs = predictions[top_indices]
    return top_probs, top_indices + 1 

def load_class_names(category_names):
    with open(category_names, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description='Predict flower name from an image using a trained model.')
    parser.add_argument('image_path', type=str, help='Path to the image.')
    parser.add_argument('model_path', type=str, help='Path to the saved Keras model.')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes.')
    parser.add_argument('--category_names', type=str, help='Path to a JSON file mapping labels to flower names.')

    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Image path {args.image_path} does not exist.")
        return
    if not os.path.exists(args.model_path):
        print(f"Model path {args.model_path} does not exist.")
        return

    model = tf.keras.models.load_model(args.model_path, custom_objects={'KerasLayer': tf.keras.layers.Layer})
    probs, classes = predict(args.image_path, model, args.top_k)

    if args.category_names:
        class_names = load_class_names(args.category_names)
        labels = [class_names.get(str(c), f"Class {c}") for c in classes]
    else:
        labels = [f"Class {c}" for c in classes]

    for prob, label in zip(probs, labels):
        print(f"{label}: {prob:.4f}")

if __name__ == '__main__':
    main()
