#!/usr/bin/env python

# Import Libraries
import keras
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Import Model
from model import create_model

def train():

	BATCH_SIZE = 32

	# Load training data
	X = np.load("../data/train_x.npy")
	label = np.load("../data/train_label.npy")
	y = to_categorical(label)

	# Create validation split
	X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=7, test_size=0.2)

	# Data augmentation
	train_datagen = ImageDataGenerator(
		rescale = 1./255,
		shear_range = 0.2,
		zoom_range = 0.2,
		horizontal_flip = True)

	val_datagen = ImageDataGenerator(rescale=1./255)

	train_gen = train_datagen.flow(X_train, y_train, batch_size = BATCH_SIZE)
	val_gen = train_datagen.flow(X_val, y_val, batch_size = BATCH_SIZE)

	# Checkpoint
	filepath = "../models/model.h5"
	checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	callbacks_list = [checkpoint]

	# Train the model
	model = create_model()
	opt = Adam(lr=0.0001)
	model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
	history = model.fit_generator(train_gen, steps_per_epoch = 608, epochs=100, validation_data=val_gen, validation_steps = 152, callbacks=callbacks_list)

	# summarize history for accuracy
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
	plt.savefig('../plots/accuracy.png')

	# summarize history for loss
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.savefig('../plots/loss.png')

if __name__ == "__main__":
	train()