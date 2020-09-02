# Python-Based Neural Network (PyBaNN)

## About
Python-Based Neural Network (PyBaNN) is my attempt to implement several neural networks encountered in [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com) in Python as I worked through the chapters.

## Requirements
- [Python 3](https://www.python.org)
- [NumPy](https://numpy.org)
- [SciPy](https://www.scipy.org)

## Usage
To run, simply type:
```shell-script
    > python3 pybann.py protocol hidden_layer epoch batch_size
```
where
- `protocol` is the neural network protocol you wish to run
- `hidden_layer` is a list of integers, specifying the number of nodes within each hidden layer (i.e. layers besides the input and output layers)
- `epoch` is the number of times to repeat the training
- During training, we break the training data into small batches and train on each of them in turn. `batch_size` is the size of these batches

The available neural network protocol which has been implemented at current are in the `network` folder. For details on each protocol, see [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com).

### Using your own protocol
You can input your own protocol. Implement your protocol as a `Network` object, with `train` and `evaluate` functions, then place your `.py` file in the `network` folder. You can then call your protocol by passing the name (excluding the `.py` extension!) of your protocol into PyBaNN.

## Neural Network Protocols
The module for implemented protocols are kept in the `network` folder.

### Protocols Implemented
- Feedforward neural network

## Datasets
All datasets can be found in the `data` folder, and all have the same structure.
- [Handwriting images](https://github.com/njhlai/homepage/blob/master/data/mnist.pkl.gz): This dataset is from the [MNIST database](http://yann.lecun.com/exdb/mnist/), adapted by Michael Nielson, the author of Neural Networks and Deep Learning. You can also get it directly [here](https://github.com/mnielsen/neural-networks-and-deep-learning).
- [Fashion article images](https://github.com/njhlai/homepage/blob/master/data/fashion_mnist.pkl.gz): This dataset is from [Zalando's article images](https://github.com/zalandoresearch/fashion-mnist), adapted by me to have the same format.