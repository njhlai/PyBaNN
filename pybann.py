import sys
import importlib
from helper import dataloader, tester

def main(protocol, networkLayers, epochs, batchSize, learningRate=1, cost='crossEntropyCost', datapath='data/mnist.pkl.gz'):
	"""initialises a network, trains on mnist database, and runs tests"""
	# training and test data
	trainingData, testData = dataloader.load(datapath)
	# initialise network
	network = importlib.import_module('network.' + protocol)
	net = network.Network(networkLayers, cost)

	# training
	print("Training a {} neural network of shape {} with {} on {}".format(protocol, networkLayers, cost, datapath))
	print("Training for {} times, each with batches of size {}".format(epochs, batchSize))
	net.train(trainingData, epochs, batchSize, learningRate)

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
	learningRate = 1 if len(sys.argv) < 6 else int(sys.argv[5])	
	cost='crossEntropyCost' if len(sys.argv) < 7 else sys.argv[6] + 'Cost'
	datapath='data/mnist.pkl.gz' if len(sys.argv) < 8 else sys.argv[7]

	# main function
	main(network, networkLayers, epochs, batchSize, learningRate, cost, datapath)