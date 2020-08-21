import sys
import importlib
import helper.mnist_loader as loader 
import helper.tester as tester

def main(protocol, networkLayers, epochs, batchSize, datapath='data/mnist.pkl.gz'):
	"""initialises a network, trains on mnist database, and runs tests"""
	# training and test data
	trainingData, testData = loader.dataLoader(datapath)
	# initialise network
	network = importlib.import_module('network.' + protocol)
	net = network.Network(networkLayers)

	# training
	print("Training a {} neural network of shape {}".format(protocol, networkLayers))
	print("Training for {} times, each with batches of size {}".format(epochs, batchSize))
	net.train(trainingData, epochs, batchSize)

	# testing
	n, correctResults = tester.test(net, testData)
	accuracy = round(correctResults / n * 100, 2)

	print("Test on {} test inputs: {} correct, giving accuracy of {}%".format(n, correctResults, accuracy))

# main function
if __name__ == "__main__":
	network = sys.argv[1]
	networkLayers = [784] + [int(s) for s in sys.argv[2].strip('][').split(',') if s != ''] + [10]
	epochs = int(sys.argv[3])
	batchSize = int(sys.argv[4])		

	main(network, networkLayers, epochs, batchSize)