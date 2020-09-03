import numpy

# Misc. math functions
def sigmoid_prime(z):
	"""derivative of sigmoid function, where z = sigmoid(x)"""
	# doing this instead of using scipy.stats.logistic._pdf reduces the accuracy, for greater speed
	return z * (1 - z)

def dummy(z):
	"""dummy function which does nothing"""
	return 1

def vectoriseLabels(y):
	"""convert labels into vector form, with 1 on the label-th position"""
	M = numpy.zeros((len(y),10))
	row = 0;

	for i in y:
		M[row, i] = 1.0
		row += 1

	return M

# cost functions
def QuadraticCost(x,y):
	"""calculation of quadratic cost"""
	return ((y - x)**2).sum() / 2

def CrossEntropyCost(x, y):
	"""calculation of cross-entropy cost"""
	return (y * numpy.log(x) + (1 - y) * numpy.log(1 - x)).sum()

# Cost class
class Cost:
	"""class for implementation of cost functions"""
	def __init__(self, method):
		"""initialise method to specify cost function"""
		self.eval = method
		self.modifier = sigmoid_prime if method == QuadraticCost else dummy

	def deltaInit(self, predicted, actual):
		"""returns initial delta appropriately according to cost function"""
		return (predicted - vectoriseLabels(actual)) * self.modifier(predicted)