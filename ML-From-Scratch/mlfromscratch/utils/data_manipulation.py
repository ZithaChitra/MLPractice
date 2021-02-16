from __future__ import division
from itertools import compinations_with_replacements
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























































































































































