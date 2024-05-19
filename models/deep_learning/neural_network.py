import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights
        self.weights_input_hidden = np.random.rand(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.rand(self.hidden_size, self.output_size)
        
        # Initialize biases
        self.bias_hidden = np.random.rand(1, self.hidden_size)
        self.bias_output = np.random.rand(1, self.output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        self.hidden = self.sigmoid(np.dot(X, self.weights_input_hidden) + self.bias_hidden)
        self.output = self.sigmoid(np.dot(self.hidden, self.weights_hidden_output) + self.bias_output)
        return self.output

    def train(self, X, y, learning_rate, epochs):
        for _ in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Compute the error
            error = y - output
            
            # Backpropagation
            d_output = error * self.sigmoid_derivative(output)
            error_hidden = d_output.dot(self.weights_hidden_output.T)
            d_hidden = error_hidden * self.sigmoid_derivative(self.hidden)
            
            # Update weights and biases
            self.weights_hidden_output += self.hidden.T.dot(d_output) * learning_rate
            self.weights_input_hidden += X.T.dot(d_hidden) * learning_rate
            self.bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
            self.bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

    def predict(self, X):
        return self.forward(X)

    def accuracy(self, X, y):
        predictions = self.predict(X) > 0.5
        return np.mean(predictions == y)
