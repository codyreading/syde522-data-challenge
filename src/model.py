# Import Keras
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

def create_model():
	numClasses = 20
	model = Sequential()

	model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(168, 308, 3)))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(numClasses, activation='softmax'))
	model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

	return model
