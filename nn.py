#!/usr/bin/env python
import numpy as np


class NeuralNetwork:
    """
    Models a simple neural network.
    """
    def __init__(self, input_units, hidden_units, output_units, output_scaling_factor):
        # network structure
        self.input_units = input_units
        self.hidden_units = hidden_units
        self.output_units = output_units

        self.output_scaling_factor = output_scaling_factor

        # weight matrix from input layer to hidden layer
        self.W1 = np.random.randn(self.input_units, self.hidden_units)
        # weight matrix from hidden layer to output layer
        self.W2 = np.random.randn(self.hidden_units, self.output_units)

        self.hidden_sum = None
        self.output_sum = None

    # TODO: Try rectified linear function
    # http://ufldl.stanford.edu/tutorial/supervised/MultiLayerNeuralNetworks/
    def sigmoid(self, x):
        """Activation function"""
        return 1 / (1 + np.exp(-x))

    def sigmoid_prime(self, x):
        """Derivative of activation function, for gradient descent"""
        return x * (1 - x)

    def forward(self, inputs):
        """Forward propagation"""
        self.hidden_sum = self.sigmoid(np.dot(inputs, self.W1))
        self.output_sum = self.sigmoid(np.dot(self.hidden_sum, self.W2))

    def backward(self, training_input, training_output):
        """Backpropagation"""
        output_error = training_output - self.output_sum
        delta_output_sum = output_error * self.sigmoid_prime(self.output_sum)

        hidden_error = delta_output_sum.dot(self.W2.T)
        delta_hidden_sum = hidden_error * self.sigmoid_prime(self.hidden_sum)

        # adjust weights
        self.W1 += training_input.T.dot(delta_hidden_sum)
        self.W2 += self.hidden_sum.T.dot(delta_output_sum)

    def epoch(self, training_input, training_output):
        self.forward(training_input)
        self.backward(training_input, training_output)

    def mean_squared_error(self, target, output_sum):
        return np.mean(np.square(target - output_sum))

    def train(self, training_input, training_output, epochs):
        print(f'normalized training input\n{training_input}')
        print(f'normalized training output\n{training_output}')
        print(f'scaled training output\n{training_output * self.output_scaling_factor}\n')

        for epoch in range(epochs):
            if epoch % 10 == 0:
                print(f'epoch {epoch}')

                self.epoch(training_input, training_output)
                print(f'output sum\n{self.output_sum}')
                print(f'scaled output sum\n{self.output_sum * self.output_scaling_factor}')

                mse = self.mean_squared_error(training_output, self.output_sum)
                print(f'mse: {mse}\n')
            else:
                self.epoch(training_input, training_output)

    def test(self, test_input):
        print(f'normalized test input: {test_input}')

        self.forward(test_input)
        print(f'output sum: {self.output_sum}')
        print(f'scaled output sum: {self.output_sum * self.output_scaling_factor}')

    # def dump_weights(self):
    #     np.savetxt('w1.txt', self.W1, fmt='%s')
    #     np.savetxt('w2.txt', self.W2, fmt='%s')


def scores(epochs):
    """
    Predict exam scores given hours studied and hours slept.
    """
    # hours studied, hours slept
    training_input = np.array(
        [
            [2, 9],
            [1, 5],
            [3, 6],
        ],
        dtype=float
    )

    # exam scores
    training_output = np.array(
        [
            [92],
            [86],
            [89],
        ],
        dtype=float
    )

    # Normalize units by dividing by the maximum value for each variable.
    # Inputs are in hours, outputs are exam scores out of 100.
    normalized_input = training_input / np.amax(training_input, axis=0)
    normalized_output = training_output / 100

    nn = NeuralNetwork(2, 3, 1, 100)
    nn.train(normalized_input, normalized_output, epochs)

    test_input = np.array(
        [
            [4, 8],
        ],
        dtype=float
    )
    normalized_test_input = test_input / np.amax(test_input, axis=0)
    nn.test(normalized_test_input)


def digits(epochs):
    """
    Digit classification
    """
    zero = [
        0, 1, 1, 0,
        1, 0, 0, 1,
        1, 0, 0, 1,
        1, 0, 0, 1,
        0, 1, 1, 0,
    ]

    one = [
        0, 0, 1, 0,
        0, 0, 1, 0,
        0, 0, 1, 0,
        0, 0, 1, 0,
        0, 0, 1, 0,
    ]

    two = [
        0, 1, 1, 0,
        1, 0, 0, 1,
        0, 0, 1, 0,
        0, 1, 0, 0,
        1, 1, 1, 1,
    ]

    three = [
        1, 1, 1, 1,
        0, 0, 0, 1,
        0, 1, 1, 1,
        0, 0, 0, 1,
        1, 1, 1, 1,
    ]

    training_input = np.array([zero, one, two, three], dtype=float)
    # TODO: Use a one-hot encoding and cross entropy instead of mse
    # to turn this into classification instead of regression.
    training_output = np.array([[0], [1], [2], [3]], dtype=float)

    # normalize units
    normalized_output = training_output / 3

    nn = NeuralNetwork(20, 20, 1, 3)
    nn.train(training_input, normalized_output, epochs)

    test = [
        0, 1, 1, 0,
        1, 0, 0, 1,
        0, 0, 1, 0,
        0, 1, 0, 0,
        1, 1, 1, 1,
    ]

    test_input = np.array([test], dtype=float)
    nn.test(test_input)
