import sys
import importlib
from helper import dataloader, tester, costfunctions

def main(protocol, networkLayers, epochs, batchSize, learningRate=1, regularisation=0, cost='cross-entropy', datapath='data/mnist.pkl.gz'):
	"""initialises a network, trains on mnist database, and runs tests"""
	# training and test data
	trainingData, testData = dataloader.load(datapath)
	# clean name of cost function
	costMethod = ''.join(x.capitalize() for x in cost.split('-'))  + 'Cost'
	if costMethod not in dir(costfunctions):
		print("WARNING: {} cost function not implemented! Reverting to cross-entropy cost.".format(cost))
		cost='cross-entropy'
		costMethod ='CrossEntropyCost'
	# initialise network
	network = importlib.import_module('network.' + protocol)
	net = network.Network(networkLayers, costfunctions.Cost(getattr(costfunctions, costMethod)))

	# training
	print("Training a {} neural network of shape {}".format(protocol, networkLayers))
	print("\t using {} cost at rate {} and regularisation factor {}".format(cost, learningRate, regularisation))
	print("\t on data: {}".format(datapath))
	print("Training for {} times, each with batches of size {}".format(epochs, batchSize))
	net.train(trainingData, epochs, batchSize, learningRate, regularisation)

	# testing
	n, correctResults = tester.test(net, testData)
	accuracy = round(correctResults / n * 100, 2)

	print("Test on {} test inputs: {} correct, giving accuracy of {}%".format(n, correctResults, accuracy))

# main function
if __name__ == "__main__":
	# parse commandline arguments
	network = sys.argv[1]
	networkLayers = [784] + [int(s) for s in sys.argv[2].strip('][').split(',') if s != ''] + [10]
	epochs = int(sys.argv[3])
	batchSize = int(sys.argv[4])
	learningRate = 1 if len(sys.argv) < 6 else float(sys.argv[5])
	regularisation = 0 if len(sys.argv) < 7 else float(sys.argv[6])
	cost='cross-entropy' if len(sys.argv) < 8 else sys.argv[7]
	datapath='data/mnist.pkl.gz' if len(sys.argv) < 9 else sys.argv[8]

	# main function
	main(network, networkLayers, epochs, batchSize, learningRate, regularisation, cost, datapath)