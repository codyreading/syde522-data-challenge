#!/usr/bin/env python

# Import Libraries
from keras.models import load_model
import pandas as pd
import numpy as np

def test():
	# Load the model
	model = load_model("../models/model.h5")

	# Load testing data
	X_test = np.load("../data/test_x.npy")

	# Predict classes
	y_test = model.predict_classes(X_test)

	# Output to CSV
	submission = pd.read_csv("../data/sample_submission.csv")
	submission['Predicted'] = y_test
	submission.to_csv("../data/submission.csv", header=True, index=False)

if __name__ == "__main__":
	test()