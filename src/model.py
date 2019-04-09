# Import Keras
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
#from keras.layers import Conv2D, MaxPooling2D, Activation
#from keras.layers.normalization import BatchNormalization
from keras.applications.resnet50 import ResNet50

def create_model():
    num_classes = 20
    HEIGHT = 168
    WIDTH = 308

    # Create ResNet50 base with all non-trainable layers
    base_model = ResNet50(weights=None, include_top=False, input_shape=(HEIGHT, WIDTH, 3), pooling = 'avg')

    x = base_model.output
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=x)
    return model