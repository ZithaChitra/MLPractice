import numpy as np

# Collection of activation functions
# Reference: http://en.wikipedia.org/wiki/Activation_function

class Sigmoid():
	def __call__(self, x):
		return 1 / (1 + np.exp(-x))

	def gradient(self, x):
		return self.__call__(x) * (1 - self.__call__(x))


class Softmax():
	def __call__(self, x):
		e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
		return e_x / np.sum(e_x, axis=1, keepdims=True)

	def gradient(self, x):
		p = self.__call__(x)
		return p * (1-p)


class TanH():
	def __call__(self, x):
		return 2 / (1 + np.exp(-2*x)) - 1

	def gradient(self, x):
		return 1 - np.power(self.__call__(x), 2)


class ReLU():
	def __call__(self, x):
		return np.where(x >= 0, x, 0)


	def gradient(self, x):
		return np.where(x >= 0, 1, 0)

	





