from __future__ import print_function, division
import math
import numpy as np
import copy
from mlfromscratch.deep_learning.activation_functions import Sigmoid, ReLU, SoftPlus, LeakyReLU
from mlfromscratch.deep_learning.activation_functions import TanH, ELU, SELU, Softmax


class Layer(object):

	def set_input_shape(self, shape):
		"""
		Sets the shape that the layers expects of the input in the forward
		pass memthod.
		"""
		self.input_shape = shape


	def layer_name(self):
		""" The name of the layer. Used in model summary. """
		return self.__class__.__name__

	def parameters(self):
		""" The number of trainable parameters used by the layer. """
		return 0

	def forward_pass(self, X, training):
		""" Propagate the signal forward in the network. """
		raise NotImplementedError()

	def backward_pass(self, accum_grad):
		"""
		Propagates the accumulated gradient backwards in the 
		network.
		If the layer has trainable weights then these weights 
		are also tuned in this method.
		As input (accum_grad) it receives the gradient with respect
		to the output of the layer and returns the gradient with
		respect to the output of the previous layer.
		"""
		raise NotImplementedError()

	def output_shape(self):
		""" The shape of the output produced by forward_pass. """
		raise NotImplementedError()


class Dense(Layer):
	"""
	A fully-connected NN layer.
	Parameters:
	----------
	n_units: int
		The number of neurons in the layers.
	input_shape: tuple
		The expected input shape of the layer. For dense layers
		a single digit specifying the number of features of the
		input. Must be specified if it is the first layer of the
		network.
	"""
	def __init__(self, n_units: int, input_shape=None):
		self.layer_input = None
		self.input_shape = input_shape
		self.n_units = n_units
		self.trainable = True
		self.W = None
		self.w0 = None

	def initialize(self, optimizer):
		# Initialize the weights
		limit = 1 / math.sqrt(self.input_shape[0])
		self.W = np.random.uniform(-limit, limit, (self.input_shape,
													self.n_units))
		self.w0 = np.zeros(1, self.n_units)
		# Weight optimizers
		self.W_opt = copy.copy(optimizer)
		self.w0_opt = copy.copy(optimizer) 
		
	def parameters(self):
		""" The number of trainable parameters used by the layer. """
		return np.prod(self.W.shape) + np.prod(self.w0.shape)

	def forward_pass(self, X, training=True):
		""" Propagates the signal forward in the network. """
		self.layer_input = X
		return X.dot(self.W) + self.w0


	def backward_pass(self, accum_grad): 
		"""
		Propagates the accumulated gradient backwards in the 
		network.
		If the layer has trainable weights then these weights 
		are also tuned in this method.
		As input (accum_grad) it receives the gradient with respect
		to the output of the layer and returns the gradient with
		respect to the output of the previous layer.
		"""
		# Save weights used during forward pass
		W = self.W

		if self.trainable:
			# Calculate gradient w.r.t layer weights
			grad_w = self.layer_input.T.dot(accum_grad)
			grad_w0 = np.sum(accum_grad, axis=0, keepdims=True)

			# update layer weights 
			self.W = self.W_opt.update(self.W, grad_w)
			self.w0 = self.w0_opt.update(self.w0, grad_w0)

		# Return accumulated gradient for next layer.
		# Calculated based on the weights used during the forward
		# pass
		accum_grad = accum_grad.dot(W.T)
		return accum_grad

	def output_shape(self):
		""" The shape of the output produced by forward pass. """
		return (self.n_units, )


		





	


	

		






















































































































































































































































































































