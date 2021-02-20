from __future__ import print_function, division
import numpy as np
import math
from mlfromscratch.utils.data_manipulation import normalize, polynomial_features


class l1_regularization():
	""" Regularization for Lasso Regression """
	def __init__(self, alpha):
		self.alpha = alpha

	def __call__(self, w):
		return self.alpha * np.linalg.norm(w)


	def grad(self, w):
		return self.alpha * np.sign(w)



class l2_regularization():
	""" Regularization for Ridge regression """
	def __init__(self, alpha):
		self.alpha = alpha

	def __call__(self, w):
		return self.alpha * 0.5 * w.T.dot(w)

	def grad(self, w):
		return self.alpha * w


class l1_l2_regularization():
	""" Regularization for Elastic Net Regression"""
	def __init__(self, alpha, l1_ratio=0.5):
		self.alpha = alpha
		self.l1_ratio = l1_ratio

	def __call__(self, w):
		l1_contr = self.l1_ratio * np.linalg.norm(w)
		l2_contr = (1 - self.l1_ratio) * w
		return self.aplha * (l1_contr + l2_contr)


	def grad(self, w):
		l1_contr = self.l1_ratio * np.sign(w)
		l2_contr = (1 - self.l1_ratio) * w
		return self.alpha * (l1_contr + l2_contr)


class Regression(object):
	"""
	Base regression model. 
	Models the relationship between a scalar dependent variable y
	and the independent variables X.
	Parameters:
	-----------
	n_iterations: float
		The number of training iterations the algorithm will tune 
		the weights for.
	learning_rate: float
		The step length that will be used when updating weights.
	"""
	def __init__(self, n_iterations, learning_rate):
		self.n_iterations = n_iterations
		self.learning_rate = learning_rate


	def initialize_weights(self, n_features):
		""" Initialize weights randomly [-1/N, 1/N] """
		limit = 1 / math.sqrt(n_features)
		self.w = np.random.uniform(-limit, limit, (n_features, ))


	def fit(self, X, y):
		# insert constant ones for bias weights
		X = np.insert(X, 0, 1, axis=1)
		self.training_errors = []
		self.initialize_weigths(n_features=X.shape[1])

		# Do gradient descent for n_iterations
		for i in range(self.n_iterations):
			y_pred = X.dot(self.w)
			# Claculate l2 loss
			mse = np.mean(0.5 * (y - y_pred)**2 + self.regularization(self.w))
			self.training_errors.append(mse)
			# Gradient of l2 loss w.r.t w
			grad_w = -(y- y_pred).dot(X) + self.regularization.grad(self.w)
			# Update the weights
			self.w -= self.learning_rate * grad_w

		
	def predict(self, X):
		# Insert constant ones for bias weights
		X = np.insert(X, 0, 1, axis=1)
		y_pred = X.dot(self.w)
		return y_pred


# ---------------------------------------------------
# +++++++++++++++++++++++++++++++++++++++++++++++++++


class LinearRegression(Regression):
	"""
	Linear model.
	Parameters:
	-----------
	n_iterations: float
		The number of iterations the algorithm will tune the weights for.
	learning_rate: float
		The step length that will be used when updating weights.
	gradient: boolean
		True or False depending if gradient descent should be used when training. if false then we use batch optimization by least squares.
	"""
	def __init__(self, n_iterations=100, learning_rate=0.001, gradient_descent=True):
		self.gradient_descent = gradient_descent
		# No regularization
		self.regularization = lambda x: 0
		self.regularization.grad = lambda x: 0
		super(LinearRegression, self).__init__(n_iteratoins=n_iterations, learning_rate=learning_rate)


	def fit(self, X, y):
		# If not gradient_descent => Least squares approximation of W
		if not self.gradient_descent:
			# insert constant ones for bias weights
			X = np.insert(X, 0, 1, axis=1)
			# Calculate weights by least squares (using Moore-Penrose pseudoinverse)
			U, S, V = np.linalg.svd(X.T.dot(X))
			S = np.diag(S)
			X_sq_reg_inv = V.dot(np.linalg.pinv(S)).dot(U.T)
			self.w = X_sq_reg_inv.dot(X.T).dot(y)

		else:
			super(LinearRegression, self).fit(X, y)
		








		
		




















































































































































