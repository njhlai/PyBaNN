# Python-Based Neural Network (PyBaNN)

## About
Python-Based Neural Network (PyBaNN) is my attempt to implement several neural networks encountered in [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com) in Python as I worked through the chapters.

## Usage
This code is Python 3 (Python 3.8) compliant. To run, simply type:
```shell-script
    > python3 pybann.py protocol hidden_layer epoch batch_size
```
where
- `protocol` is the neural network protocol you wish to run
- `hidden_layer` is a list of integers, specifying the number of hidden layers (i.e. layers besides the input and output layers)
- `epoch` is the number of times to repeat the training
- In training, we break the training data into small batches and train on each of them in turn. `batch_size` is the size of these batches

The available neural network protocol which has been implemented at current are in the `network` folder. For details on each protocol, see [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com).

### Using your own protocol
You can input your own protocol. Implement your protocol as a `Network` object, with `train` and `evaluate` functions, then place your `.py` file in the `network` folder. You can then call your protocol by passing the name (excluding the `.py` extension!) of your protocol into PyBann.