from __future__ import division
from itertools import combinations_with_replacements
import numpy as np
import math
import sys


def shuffle_data(X, y, seed=None):
	""" Random shuffle of the samples in X and y """
	if seed:
		np.random.seed(seed)
	idx = np.arange(X.shape[0])
	np.random.shuffle(idx)
	return X[idx], y[idx]


def batch_iterator(X, y=None, batch_size=64):
	""" Simple batch generator """
	n_samples = X.shape[0]
	for i in np.arange(0, n_samples, batch_size):
		begin, end = i, min(i+batch_size, n_samples)
		if y is not None:
			yield X[begin:end], y[begin:end]
		else:
			yield X[begin:end]



def divide_on_feature(X, feature_i, threshold):
	"""
	 Divide dataset based on if sample value on feature index is 
	 larger than given threshold
	"""
	split_fu= None
	if isinstance(threshold, int) or isinstance(threshold, float):
		spli_func = lambda sample: sample[feature_i] >= threshold
	else:
		split_func = lambda sample: sample[feature_i] == threshold


	X_1 = np.array([sample for sample in X if split_func(sample)]) 
	X_2 = np.array([sample for sample in X if not split_func(sample)])

	return np.array([X_1, X_2])


def polynomial_features(X, degree):
	n_samples, n_features = np.shape(X)

	def index_combinations():
		combs = [combinations_with_replacements(range(n_features), 1) for i in range(0, degree + 1)]
		flat_combs = [item for sublist in combs for item in sublist]
		return flat_combs

	combinations = index_combinations()
	n_output_features = len(combinations)
	X_new = np.empty(n_samples, n_output_features)

	for i, index_combs in enumerate(combinations):
		X_new[:, i] = np.prod(X[:, index_combs], axis=1)

	return X_new


	def get_random_subset(X, y):
		



























































































































































