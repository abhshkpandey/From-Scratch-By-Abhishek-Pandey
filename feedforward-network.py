import random
import math

class Neuron:
    def __init__(self, num_inputs):
        self.weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
        self.bias = random.uniform(-1, 1)
    
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
    
    def sigmoid_derivative(self, x):
        sig = self.sigmoid(x)
        return sig * (1 - sig)
    
    def forward(self, inputs):
        if len(inputs) != len(self.weights):
            raise ValueError(f"Expected {len(self.weights)} inputs, got {len(inputs)}")
        
        self.last_inputs = inputs
        self.weighted_sum = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        self.output = self.sigmoid(self.weighted_sum)
        return self.output

class Layer:
    def __init__(self, num_neurons, num_inputs):
        self.neurons = [Neuron(num_inputs) for _ in range(num_neurons)]
    
    def forward(self, inputs):
        self.outputs = [neuron.forward(inputs) for neuron in self.neurons]
        return self.outputs

class NeuralNetwork:
    def __init__(self, layer_sizes):
        """
        layer_sizes: list of integers representing the number of neurons in each layer
        Example: [2, 3, 1] creates a network with 2 inputs, 3 hidden neurons, and 1 output
        """
        self.layers = []
        for i in range(1, len(layer_sizes)):
            layer = Layer(layer_sizes[i], layer_sizes[i-1])
            self.layers.append(layer)
    
    def forward(self, inputs):
        """Forward propagation through the network"""
        current_inputs = inputs
        for layer in self.layers:
            current_inputs = layer.forward(current_inputs)
        return current_inputs
    
    def backpropagation(self, inputs, target, learning_rate=0.1):
        """Train the network using backpropagation"""
        # Forward pass
        layer_inputs = [inputs]
        current_inputs = inputs
        
        for layer in self.layers:
            current_inputs = layer.forward(current_inputs)
            layer_inputs.append(current_inputs)
        
        # Calculate output layer error
        output_errors = []
        output_layer = self.layers[-1]
        
        for i, neuron in enumerate(output_layer.neurons):
            error = target[i] - neuron.output
            delta = error * neuron.sigmoid_derivative(neuron.weighted_sum)
            output_errors.append(delta)
        
        # Backpropagate error
        layer_errors = [output_errors]
        for i in range(len(self.layers)-2, -1, -1):
            layer = self.layers[i]
            next_layer = self.layers[i+1]
            layer_error = []
            
            for j, neuron in enumerate(layer.neurons):
                error = 0
                for k, next_neuron in enumerate(next_layer.neurons):
                    error += next_neuron.weights[j] * layer_errors[-1][k]
                delta = error * neuron.sigmoid_derivative(neuron.weighted_sum)
                layer_error.append(delta)
            
            layer_errors.append(layer_error)
        
        layer_errors.reverse()
        
        # Update weights and biases
        for i, layer in enumerate(self.layers):
            inputs = layer_inputs[i]
            for j, neuron in enumerate(layer.neurons):
                delta = layer_errors[i][j]
                for k in range(len(inputs)):
                    neuron.weights[k] += learning_rate * delta * inputs[k]
                neuron.bias += learning_rate * delta
    
    def train(self, training_data, epochs, learning_rate=0.1):
        """Train the network on a set of training data"""
        for epoch in range(epochs):
            total_error = 0
            for inputs, targets in training_data:
                # Forward pass
                outputs = self.forward(inputs)
                
                # Calculate error
                error = sum((t - o) ** 2 for t, o in zip(targets, outputs))
                total_error += error
                
                # Backpropagation
                self.backpropagation(inputs, targets, learning_rate)
            
            # Print progress
            if (epoch + 1) % 1000 == 0:
                print(f"Epoch {epoch + 1}, Error: {total_error:.4f}")

def main():
    # Create a neural network with 2 inputs, 3 hidden neurons, and 1 output
    network = NeuralNetwork([2, 3, 1])
    
    # Training data for XOR gate
    training_data = [
        ([0, 0], [0]),
        ([0, 1], [1]),
        ([1, 0], [1]),
        ([1, 1], [0])
    ]
    
    # Train the network
    print("Training network...")
    network.train(training_data, epochs=10000, learning_rate=0.1)
    
    # Test the trained network
    print("\nTesting network:")
    for inputs, targets in training_data:
        outputs = network.forward(inputs)
        print(f"Inputs: {inputs}, Target: {targets[0]}, Output: {outputs[0]:.4f}")

if __name__ == "__main__":
    main()
