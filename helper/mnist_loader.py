import gzip
import pickle
import numpy

def dataLoader(path='data/mnist.pkl.gz'):
	"""mnist data loader, outputs (combined) training and test data
		variable: path='data/mnist.pkl.gz'"""
	with gzip.open(path, 'rb') as f:
		training, validation, test = pickle.load(f, encoding='latin1')

	# each data variable is a tuple (M, l), where 
	# 	M is matrix where each row vector is an array of pixel inputs
	# 	l is an array of labels

	# concatenate the training and validation into one, using zip iteration
	combTraining = tuple(numpy.concatenate((t,v), axis=0) for t,v in zip(training, validation))

	# return (combined) training and test data
	return (combTraining, test)