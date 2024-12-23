import os
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.model import build_model

# Paths
DATA_DIR = "/content/drive/MyDrive/Data_mining"  # Update with your dataset path
MODEL_SAVE_PATH = "./models/multi_attribute_model.h5"

# Load and preprocess data
def load_data():
    data = pd.read_csv(os.path.join(DATA_DIR, "data.csv"))
    x_data = []
    y_gender, y_age, y_race = [], [], []
    
    for _, row in data.iterrows():
        img_path = os.path.join(DATA_DIR, "images", row["image"])
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
        img = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        x_data.append(img)
        
        y_gender.append(row["gender"])
        y_age.append(row["age"])
        y_race.append(row["race"])

    x_data = np.array(x_data)
    y_gender = np.array(y_gender)
    y_age = np.array(y_age)
    y_race = to_categorical(y_race, num_classes=5)

    return x_data, y_gender, y_age, y_race

# Main training function
def train():
    # Load data
    x_data, y_gender, y_age, y_race = load_data()
    x_train, x_val, y_gender_train, y_gender_val, y_age_train, y_age_val, y_race_train, y_race_val = train_test_split(
        x_data, y_gender, y_age, y_race, test_size=0.2, random_state=42
    )

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    datagen.fit(x_train)

    # Build model
    model = build_model()
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss={
            "gender_output": "binary_crossentropy",
            "age_output": "mse",
            "race_output": "categorical_crossentropy"
        },
        metrics={
            "gender_output": "accuracy",
            "age_output": "mae",
            "race_output": "accuracy"
        }
    )

    # Train model
    history = model.fit(
        datagen.flow(x_train, {"gender_output": y_gender_train, "age_output": y_age_train, "race_output": y_race_train}),
        validation_data=(x_val, {"gender_output": y_gender_val, "age_output": y_age_val, "race_output": y_race_val}),
        epochs=20,
        batch_size=32
    )

    # Save the trained model
    os.makedirs("./models", exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved at {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()
