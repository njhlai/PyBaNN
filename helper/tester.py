import numpy

# testing function
def test(network, testData):
	"""tests network with testData"""
	testInputs, testLabels = testData
	testOutputs = network.evaluate(testInputs)

	# converts outputs into labels: the index with the highest value is the label
	outputLabels = numpy.argmax(testOutputs[-1], axis=1)
	
	return len(testLabels), sum(int(output == label) for output, label in zip(outputLabels, testLabels))