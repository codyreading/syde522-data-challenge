#!/usr/bin/env python

# Import Libraries
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input

import pandas as pd
import numpy as np

def test():
	# Load the model
	model = load_model("../models/model.h5")

	# Load testing data
	X_test = np.load("../data/test_x.npy")

	# Data preprocessing
	test_datagen = ImageDataGenerator(
		rescale=1./255,
		preprocessing_function=preprocess_input)

	test_gen = test_datagen.flow(X_test, batch_size=1)

	# Predict classes
	y_prob = model.predict_generator(test_gen, steps = 200)
	y_test = y_prob.argmax(axis=-1)

	import pdb;pdb.set_trace()

	# Output to CSV
	submission = pd.read_csv("../data/sample_submission.csv")
	submission['Predicted'] = y_test
	submission.to_csv("../data/submission.csv", header=True, index=False)

if __name__ == "__main__":
	test()