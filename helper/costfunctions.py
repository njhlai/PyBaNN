import numpy

# Misc. math functions
def sigmoid_prime(z):
	"""derivative of sigmoid function, where z = sigmoid(x)"""
	# doing this instead of using scipy.stats.logistic._pdf reduces the accuracy, for greater speed
	return z * (1 - z)

def dummy(z):
	return 1

# cost functions
def quadraticCost(x,y):
	"""calculation of average quadratic cost"""
	return ((y - x)**2).sum() / 2

def crossEntropyCost(x, y):
	"""calculation of average cross-entropy cost"""
	return (y * numpy.log(x) + (1 - y) * numpy.log(1 - x)).sum()

# Cost class
class Cost:
	"""class for implementation of cost functions"""
	def __init__(self, method):
		"""initialise method to specify cost function"""
		self.eval = method
		self.modifier = sigmoid_prime if method == quadraticCost else dummy

	def deltaInit(self, predicted, actual):
		"""returns initial delta appropriately according to cost function"""
		return ((predicted - actual) * self.modifier(predicted)).transpose()