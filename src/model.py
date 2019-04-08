# Import Keras
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras.layers.normalization import BatchNormalization
from keras.applications.resnet50 import ResNet50

def create_model():
    num_classes = 20
    HEIGHT = 168
    WIDTH = 308

    # Create ResNet50 base with all non-trainable layers
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(num_classes, activation='softmax')(x)
    model = Model(base_model.input, x)
    model.save_weights("../models/all_nontrainable.h5")

    base_model = ResNet50(include_top=False, weights="imagenet", input_shape=(HEIGHT, WIDTH, 3))
    for layer in base_model.layers[:-10]:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=x)
    model.load_weights("../models/all_nontrainable.h5")
    return model