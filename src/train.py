#!/usr/bin/env python

# Import Libraries
import keras
from keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split

# Import Model
from model import create_model

def train():
	# Load training data
	X = np.load("../data/train_x.npy")
	label = np.load("../data/train_label.npy")
	y = to_categorical(label)

	# Create validation split
	X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=7, test_size=0.2)

	# Train the model
	model = create_model()
	model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

	# Save the model
	model.save("../models/model.h5")

if __name__ == "__main__":
	train()