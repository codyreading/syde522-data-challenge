# Import Keras
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

class model(Sequential):
	def __init__(self):
		numClasses = 20

		self.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(28,28,1)))
		self.add(Conv2D(64, (3, 3), activation='relu'))
		self.add(MaxPooling2D(pool_size=(2, 2)))
		self.add(Dropout(0.25))
		self.add(Flatten())
		self.add(Dense(128, activation='relu'))
		self.add(Dropout(0.5))
		self.add(Dense(numClasses, activation='softmax'))

		self.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
