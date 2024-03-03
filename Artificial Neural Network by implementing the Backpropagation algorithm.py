import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
        self.bias_output = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        # Forward pass
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = self.sigmoid(self.final_input)
        return self.final_output

    def backward(self, X, y, output):
        # Backpropagation
        error = y - output
        output_delta = error * self.sigmoid_derivative(output)

        error_hidden = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = error_hidden * self.sigmoid_derivative(self.hidden_output)

        # Update weights and biases
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta)
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True)
        self.weights_input_hidden += X.T.dot(hidden_delta)
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True)

    def train(self, X, y, epochs):
        loss_history = []
        for epoch in range(epochs):
            # Forward and backward pass for each training example
            for i in range(len(X)):
                input_data = X[i:i+1]
                target = y[i:i+1]

                output = self.forward(input_data)
                self.backward(input_data, target, output)

            # Calculate and record loss every epoch
            loss = np.mean(np.square(y - self.forward(X)))
            loss_history.append(loss)

            # Print loss every 100 epochs
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')

        return loss_history

# Example usage with a simple dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input
y = np.array([[0], [1], [1], [0]])  # Output

# Create a neural network with 2 input neurons, 2 hidden neurons, and 1 output neuron
nn = NeuralNetwork(input_size=2, hidden_size=2, output_size=1)

# Train the neural network for 1000 epochs
loss_history = nn.train(X, y, epochs=1000)

# Test the trained network
test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
predictions = nn.forward(test_data)
print("Predictions:")
print(predictions)

# Plot the network architecture
def plot_neural_network(input_size, hidden_size, output_size):
    plt.figure(figsize=(8, 6))
    layer_sizes = [input_size, hidden_size, output_size]
    layer_labels = ['Input Layer', 'Hidden Layer', 'Output Layer']
    y_positions = [3, 2, 1]
    
    for i in range(len(layer_sizes) - 1):
        layer_label = layer_labels[i]
        for j in range(layer_sizes[i]):
            plt.scatter(i, y_positions[j], color='blue')
            plt.text(i, y_positions[j], f'{layer_label}\nNeuron {j+1}', fontsize=12, ha='center', va='center')
            
        for k in range(layer_sizes[i + 1]):
            plt.scatter(i + 1, y_positions[k], color='red')
            plt.text(i + 1, y_positions[k], f'{layer_labels[i + 1]}\nNeuron {k+1}', fontsize=12, ha='center', va='center')
            
        for j in range(layer_sizes[i]):
            for k in range(layer_sizes[i + 1]):
                plt.plot([i, i + 1], [y_positions[j], y_positions[k]], color='black')
    
    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.title('Neural Network Architecture')
    plt.show()

plot_neural_network(input_size=2, hidden_size=2, output_size=1)

# Display summary report
summary_data = {'Epoch': range(len(loss_history)), 'Loss': loss_history}
summary_df = pd.DataFrame(summary_data)
print("\nSummary Report:")
print(summary_df)
