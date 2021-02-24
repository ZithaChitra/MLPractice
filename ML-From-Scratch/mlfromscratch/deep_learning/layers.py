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

	

		






















































































































































































































































































































