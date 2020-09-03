import numpy
from scipy.special import expit # more efficient calculation of sigmoid

# from helper import costfunctions

# Misc. functions
def sigmoid_prime(z):
	"""derivative of sigmoid function, where z = sigmoid(x)"""
	# doing this instead of using scipy.stats.logistic._pdf reduces the accuracy, for greater speed
	return z * (1 - z)

def regularisationCost(w):
	"""calculation of L2 regularisation"""
	return (w**2).sum() / 2

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

	def __init__(self, sizes, cost):
		"""initialise weights and biases for the feedforward neural network, along with specified cost"""
		assert sizes[0] == 784 # assert that the first layer has 784 = 28x28 nodes, one for each pixel
		assert sizes[-1] == 10 # assert that the last layer has 10 nodes, one for each digit

		self.layerNum = len(sizes)
		self.weights = [numpy.random.randn(input_dim+1, output_dim) for output_dim,input_dim in zip(sizes[1:], sizes[:-1])]
		self.cost = cost

	def averageCost(self, x, y, regularisation=0):
		"""calculation of average cost"""
		n = len(x)
		assert n == len(y)

		# calculate regularisation cost
		regularisationCost = 0
		if regularisation:
			for w in self.weights:
				regularisationCost = regularisation * L2Regularisation(w)

		return (self.cost.eval(x, y) + regularisationCost) / n

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
		ones = numpy.ones((1,len(inputs)))

		# initialise variables
		# # delta, initialised with (output - label) \odot modifier (see costfunctions.Cost class)
		delta = self.cost.deltaInit(outputs[-1], labels)
		# nabla
		nabla = []

		# iterating backwards
		for i in range(1, self.layerNum):
			# nabla = [previous layer outputs]^T * delta
			nabla = [numpy.concatenate((outputs[-i-1].transpose(), ones), axis=0).dot(delta)] + nabla

			# [previous layer delta] = 
			# ([current layer delta] * [current layer weights excluding biases]^T) \odot [previous layer sigmoid_prime]
			delta = delta.dot(self.weights[-i][:-1].transpose()) * sigmoid_prime(outputs[-i-1])

		return nabla 

	def updateWeights(self, inputs, labels, total, learningRate=1, regularisation=0):
		"""updates the weights of the network by gradient descent"""
		assert len(inputs) == len(labels)
		n = len(inputs)

		# backpropagate
		nabla = self.backpropagation(inputs, labels)

		# calculated new weight
		for W,V in zip(self.weights, nabla):
			# biases are ignored when tweaking according to regularisation
			if regularisation: W[:-1,:] *= 1 - ((learningRate * regularisation) / total)
			# new weight = regularised old weight - \frac{learningRate}{n} nabla
			W -= (learningRate  * V) / n

	def train(self, trainingData, epoch, batchSize, learningRate=1, regularisation=0):
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
				self.updateWeights(inputs[beg:end], labels[beg:end], n, learningRate, regularisation)
				beg = end
				print("Epoch {}: trained {} entries".format(i, end), end="\r", flush=True)

			print("Epoch {}: [DONE]                ".format(i), end="\r", flush=True)