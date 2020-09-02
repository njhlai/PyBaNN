import numpy
from scipy.special import expit # more efficient calculation of sigmoid

# Misc. math functions
def sigmoid(x):
	"""sigmoid function"""
	return 1 / (1 + numpy.exp(-x))

def sigmoid_prime(z):
	"""derivative of sigmoid function, where z = sigmoid(x)"""
	# doing this instead of using scipy.stats.logistic._pdf reduces the accuracy, for greater speed
	return z * (1 - z)

def vectoriseLabels(y):
	"""convert labels into vector form, with 1 on the label-th position"""
	M = numpy.zeros((len(y),10))
	row = 0;

	for i in y:
		M[row, i] = 1.0
		row += 1

	return M

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

def averageCost(x,y):
	"""calculation of average cost via average mean square error"""
	assert len(x) == len(y)
	n = len(x)
	return 1 / (2 * n) * (x - y)**2

# Network class
class Network:
	"""class for Feedforward Neural Network implementation"""

	def __init__(self, sizes):
		"""initialise weights and biases for the feedforward neural network"""
		assert sizes[0] == 784 # assert that the first layer has 784 = 28x28 nodes, one for each pixel
		assert sizes[-1] == 10 # assert that the last layer has 10 nodes, one for each digit

		self.layerNum = len(sizes)
		self.weights = [numpy.random.randn(input_dim+1, output_dim) for output_dim,input_dim in zip(sizes[1:], sizes[:-1])]

	def evaluate(self, inputs):
		"""evaluation of multilayer (continuous) perceptron network"""

		# output of each layer
		outputs = [inputs]
		# activation of each layer
		derivatives_t = []
		# column vector of 1s
		ones = numpy.ones((len(inputs), 1))

		for M in self.weights:
			# add one more input (column) dimension with value 1 for bias
			currentActivation = numpy.concatenate((outputs[-1], ones), axis=1)

			# dot product
			currentActivation = currentActivation.dot(M)

			# sigmoid and sigmoid' calculation
			outputs += [expit(currentActivation)]
			derivatives_t += [sigmoid_prime(outputs[-1]).transpose()]

		return derivatives_t, outputs

	def backpropagation(self, inputs, labels):
		"""backpropogation algorithm, returning the gradient of the cost function"""
		derivatives_t, outputs = self.evaluate(inputs)

		# row vector of 1s
		ones = numpy.ones((len(inputs),1))

		# initialise variables
		# delta^T, initialised with (output - label)
		delta_t = (outputs[-1] - vectoriseLabels(labels)).transpose()
		# multiplier at each step to delta^T, initialised with identity matrix
		multiplier = numpy.identity(10) # it will be the matrix of weights from the next layer
		# nabla
		nabla = []

		# iterating backwards
		for i in range(1, self.layerNum):
			# delta^T = (weights * [previous delta]) \odot sigmoid'^T
			delta_t = multiplier.dot(delta_t) * derivatives_t[-i]

			# nabla = (delta^T * [previous layer outputs])^T
			nabla = [delta_t.dot(numpy.concatenate((outputs[-i-1], ones), axis=1)).transpose()] + nabla

			# next multiplier is the weights of the current matrix, excluding biases
			multiplier = self.weights[-i][:-1]

		return nabla 

	def updateWeights(self, inputs, labels):
		"""updates the weights of the network by gradient descent"""
		assert len(inputs) == len(labels)

		nabla = self.backpropagation(inputs, labels)
		n = len(inputs)

		# new weight = old weight - \frac{nabla}{n}
		self.weights = [W - 1/n * V for W,V in zip(self.weights, nabla)]

	def train(self, trainingData, epoch, batchSize):
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
				self.updateWeights(inputs[beg:end], labels[beg:end])
				beg = end
				print("Epoch {}: trained {} entries".format(i, end), end="\r", flush=True)

			print("Epoch {}: [DONE]                ".format(i))