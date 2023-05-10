import os
import cv2
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set image and data parameters
IMAGE_SIZE = 100
BATCH_SIZE = 32
EPOCHS = 50
IMAGES_DIR = "output"
MODEL_FILE = "trained_model.h5"


if __name__ == "__main__":
    # Load and preprocess images
    print("Loading and preprocessing images...")
    image_data = []
    image_labels = []

    for idx, category in enumerate(os.listdir(IMAGES_DIR)):
        if 'please_click_the_center' in category:
            print(f"Skipping category: {category}")
            continue
        print(f"Processing category: {category}")
        category_dir = os.path.join(IMAGES_DIR, category, "yes")
        for image_file in os.listdir(category_dir):
            image_path = os.path.join(category_dir, image_file)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
            image_data.append(image)
            image_labels.append(idx)

    image_data = np.array(image_data) / 255.0
    image_labels = np.array(image_labels)

    # Shuffle the data
    p = np.random.permutation(len(image_data))
    image_data = image_data[p]
    image_labels = image_labels[p]

    # Create data generator
    data_gen = ImageDataGenerator(validation_split=0.2)
    train_data_gen = data_gen.flow(image_data, image_labels, batch_size=BATCH_SIZE, subset='training')
    val_data_gen = data_gen.flow(image_data, image_labels, batch_size=BATCH_SIZE, subset='validation')

    # Create and compile the model
    print("Creating and compiling the model...")
    base_model = MobileNetV2(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, weights='imagenet')
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(len(os.listdir(IMAGES_DIR)), activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    print("Training the model...")
    model.fit(train_data_gen, epochs=EPOCHS, validation_data=val_data_gen)

    # Save the trained model
    print("Saving the trained model...")
    model.save(MODEL_FILE)
    print("Model training and saving completed.")
