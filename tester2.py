import os
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import random

from trainer2 import IMAGE_SIZE, IMAGES_DIR, MODEL_FILE


def classify_image(image_path, model):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)

    categories = os.listdir(IMAGES_DIR)
    return categories[predicted_class], prediction


model = load_model(MODEL_FILE)
category = 'electric_scooter'
data_type = 'yes'  # yes / bad
potential_images = os.listdir(os.path.join(IMAGES_DIR, category, data_type))

for _ in range(10):
    input_image = os.path.join(IMAGES_DIR, category, data_type, random.choice(potential_images))
    result, prediction = classify_image(input_image, model)
    print(f"The image is classified as: {result}")
    #print(f"Prediction probabilities: {prediction}")
    print("ERROR" if result != category else "SUCCESS")

    # If you want to show the top N alternatives
    N = 3  # You can change this to the desired number of alternatives
    sorted_categories = sorted(zip(os.listdir(IMAGES_DIR), prediction[0]), key=lambda x: x[1], reverse=True)
    top_n_alternatives = sorted_categories[:N]
    print(f"Top {N} alternatives: {top_n_alternatives}")
