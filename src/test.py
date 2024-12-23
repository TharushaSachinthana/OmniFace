import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Paths
MODEL_PATH = "./models/multi_attribute_model.h5"
TEST_IMAGE_PATH = "/content/drive/MyDrive/vision/Usain_Bolt_portrait.jpg"  # Update with your custom image path

# Labels
gender_dict = {0: "Male", 1: "Female"}
race_dict = {0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Other"}

# Load model
def load_trained_model():
    model = load_model(MODEL_PATH)
    return model

# Predict on custom image
def predict_on_image(image_path, model):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)

    gender_pred = predictions[0].flatten()
    age_pred = predictions[1].flatten()
    race_pred = predictions[2].flatten()

    gender = gender_dict[int(round(gender_pred[0]))]
    age = int(round(age_pred[0]))
    race = race_dict[np.argmax(race_pred)]

    print(f"Predicted Gender: {gender}")
    print(f"Predicted Age: {age}")
    print(f"Predicted Race: {race}")

if __name__ == "__main__":
    # Load the trained model
    model = load_trained_model()

    # Predict on the test image
    if os.path.exists(TEST_IMAGE_PATH):
        predict_on_image(TEST_IMAGE_PATH, model)
    else:
        print(f"Image path {TEST_IMAGE_PATH} does not exist.")
