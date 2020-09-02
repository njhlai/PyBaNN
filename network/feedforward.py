import numpy
from scipy.special import expit # more efficient calculation of sigmoid

import helper.costfunctions as costfunctions

# Misc. functions
def sigmoid_prime(z):
	"""derivative of sigmoid function, where z = sigmoid(x)"""
	# doing this instead of using scipy.stats.logistic._pdf reduces the accuracy, for greater speed
	return z * (1 - z)

def unisonShuffle(a,b):
	"""shuffles two numpy arrays of the same length in unison"""
	assert len(a) == len(b)

	# get random state
	rng_state = numpy.random.get_state()
	# shuffle a
	numpy.random.shuffle(a)
	# reset random state to rng_state
	numpy.random.set_state(rng_state)
	# shuffle b with the same permutation (determined by random state)
	numpy.random.shuffle(b)

# Network class
class Network:
	"""class for Feedforward Neural Network implementation"""

	def __init__(self, sizes, cost='crossEntropyCost'):
		"""initialise weights and biases for the feedforward neural network, along with specified cost"""
		assert sizes[0] == 784 # assert that the first layer has 784 = 28x28 nodes, one for each pixel
		assert sizes[-1] == 10 # assert that the last layer has 10 nodes, one for each digit

		self.layerNum = len(sizes)
		self.weights = [numpy.random.randn(input_dim+1, output_dim) for output_dim,input_dim in zip(sizes[1:], sizes[:-1])]
		self.cost = costfunctions.Cost(getattr(costfunctions, cost))

	def averageCost(self, x, y):
		n = len(x)
		assert n == len(y)

		return self.cost.eval(x, y) / n

	def evaluate(self, inputs):
		"""evaluation of multilayer (continuous) perceptron network"""

		# output of each layer
		outputs = [inputs]
		# column vector of 1s
		ones = numpy.ones((len(inputs), 1))

		for M in self.weights:
			# add one more input (column) dimension with value 1 for bias
			currentActivation = numpy.concatenate((outputs[-1], ones), axis=1)

			# dot product
			currentActivation = currentActivation.dot(M)

			# sigmoid calculation
			outputs += [expit(currentActivation)]

		return outputs

	def backpropagation(self, inputs, labels):
		"""backpropogation algorithm, returning the gradient of the cost function"""
		outputs = self.evaluate(inputs)

		# row vector of 1s
		ones = numpy.ones((len(inputs),1))

		# initialise variables
		# # delta^T, initialised with (output - label) \odot modifier (see costfunctions.Cost class)
		delta_t = self.cost.deltaInit(outputs[-1], labels).transpose()
		# nabla
		nabla = []

		# iterating backwards
		for i in range(2, self.layerNum+1):
			# nabla = (delta^T * [previous layer outputs])^T
			nabla = [delta_t.dot(numpy.concatenate((outputs[-i], ones), axis=1)).transpose()] + nabla

			# [next delta]^T = 
			# ([next layer weights excluding biases] * [current delta]^T) \odot sigmoid_prime([current layer outputs])^T
			delta_t = self.weights[-i+1][:-1].dot(delta_t) * sigmoid_prime(outputs[-i].transpose())

		return nabla 

	def updateWeights(self, inputs, labels, learningRate):
		"""updates the weights of the network by gradient descent"""
		assert len(inputs) == len(labels)

		nabla = self.backpropagation(inputs, labels)
		n = len(inputs)

		# new weight = old weight - \frac{learningRate}{n} nabla
		self.weights = [W - (learningRate  * V) / n for W,V in zip(self.weights, nabla)]

	def train(self, trainingData, epoch, batchSize, learningRate):
		"""trains the network via stochastic gradient descent, repeated 'epoch' amount of time, with 'trainingData' broken into batches of size 'batchSize'"""
		inputs, labels = trainingData
		n = len(inputs)

		for i in range(epoch):
			# shuffling inputs, labels in unison
			print("Epoch {}: shuffling dataset...".format(i), end="\r", flush=True)
			unisonShuffle(inputs, labels)
			beg = 0

			for k in range(0, n // batchSize):
				end = beg + batchSize
				self.updateWeights(inputs[beg:end], labels[beg:end], learningRate)
				beg = end
				print("Epoch {}: trained {} entries".format(i, end), end="\r", flush=True)

			print("Epoch {}: [DONE]                ".format(i))