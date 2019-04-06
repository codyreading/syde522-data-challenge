#!/usr/bin/env python

# Import Libraries
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from tqdm import tqdm

# Import Model
from model import model

def train():
	# Read the data
	X = np.load("../data/train_x.npy")
	label = np.load("../data/train_label.npy")
	y = to_categorical(label)
	import pdb; pdb.set_trace()

	# Create validation split
	X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=7, test_size=0.2)

	# Train the model
	model = model()
	model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

	# Save the model
	model.save("../models/model.h5")

if __name__ == "__main__":
	train()