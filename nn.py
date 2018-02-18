import numpy as np


class NeuralNetwork:
    """
    Models a simple neural network.
    Outputs exam score given hours studied and hours slept.
    """
    def __init__(self):
        # network structure
        self.input_units = 2
        self.hidden_units = 3
        self.output_units = 1

        # weight matrix from input layer to hidden layer
        self.W1 = np.random.randn(self.input_units, self.hidden_units)
        # weight matrix from hidden layer to output layer
        self.W2 = np.random.randn(self.hidden_units, self.output_units)

        self.hidden_sum = None
        self.output_sum = None

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
        for epoch in range(epochs):
            print(f'epoch: {epoch}')
            print(f'normalized input: {training_input}')
            print(f'target output: {training_output}')

            self.epoch(training_input, training_output)
            print(f'output sum: {self.output_sum}')

            mse = self.mean_squared_error(training_output, self.output_sum)
            print(f'mse: {mse}')

    def test(self, test_input):
        print(f'normalized input: {test_input}')

        self.forward(test_input)
        print(f'output sum: {self.output_sum}')

    # def dump_weights(self):
    #     np.savetxt('w1.txt', self.W1, fmt='%s')
    #     np.savetxt('w2.txt', self.W2, fmt='%s')


# hours studied, hours slept
training_input = np.array(
    (
        [2, 9],
        [1, 5],
        [3, 6],
    ),
    dtype=float
)

# exam score
training_output = np.array(
    (
        [92],
        [86],
        [89],
    ),
    dtype=float
)

# Normalize units by dividing by the maximum value for each variable.
# Inputs are in hours, outputs are exam scores out of 100.
normalized_input = training_input / np.amax(training_input, axis=0)
normalized_output = training_output / 100

nn = NeuralNetwork()
nn.train(normalized_input, normalized_output, 10)

test_input = np.array(([4, 8]), dtype=float)
normalized_test_input = test_input / np.amax(test_input, axis=0)
nn.test(normalized_test_input)
