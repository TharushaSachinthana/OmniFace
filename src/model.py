import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.models import Model

def build_model():
    input_layer = Input(shape=(224, 224, 3), name="input_layer")
    base_model = VGG16(weights="imagenet", include_top=False, input_tensor=input_layer)
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Shared layers
    x = Flatten(name="flatten")(base_model.output)
    
    # Gender branch
    gender_branch = Dense(256, activation="relu", name="gender_dense")(x)
    gender_branch = Dropout(0.5, name="gender_dropout")(gender_branch)
    gender_output = Dense(1, activation="sigmoid", name="gender_output")(gender_branch)

    # Age branch
    age_branch = Dense(256, activation="relu", name="age_dense")(x)
    age_branch = Dropout(0.5, name="age_dropout")(age_branch)
    age_output = Dense(1, activation="linear", name="age_output")(age_branch)

    # Race branch
    race_branch = Dense(128, activation="relu", name="race_dense")(x)
    race_branch = Dropout(0.5, name="race_dropout")(race_branch)
    race_output = Dense(5, activation="softmax", name="race_output")(race_branch)

    # Combine branches
    model = Model(inputs=input_layer, outputs=[gender_output, age_output, race_output], name="multi_attribute_model")

    return model
