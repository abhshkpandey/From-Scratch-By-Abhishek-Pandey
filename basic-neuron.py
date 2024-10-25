import random
import math

class Neuron:
    def __init__(self, num_inputs):
        # Initialize weights with random values between -1 and 1
        self.weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
        self.bias = random.uniform(-1, 1)
    
    def sigmoid(self, x):
        # Sigmoid activation function
        return 1 / (1 + math.exp(-x))
    
    def sigmoid_derivative(self, x):
        # Derivative of sigmoid function
        sig = self.sigmoid(x)
        return sig * (1 - sig)
    
    def forward(self, inputs):
        # Check if input size matches weight size
        if len(inputs) != len(self.weights):
            raise ValueError(f"Expected {len(self.weights)} inputs, got {len(inputs)}")
        
        # Calculate weighted sum
        self.last_inputs = inputs
        self.weighted_sum = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        
        # Apply activation function
        self.output = self.sigmoid(self.weighted_sum)
        return self.output
    
    def train(self, inputs, target, learning_rate=0.1):
        # Forward pass
        output = self.forward(inputs)
        
        # Calculate error
        error = target - output
        
        # Calculate gradient for output
        delta = error * self.sigmoid_derivative(self.weighted_sum)
        
        # Update weights
        for i in range(len(self.weights)):
            self.weights[i] += learning_rate * delta * self.last_inputs[i]
        
        # Update bias
        self.bias += learning_rate * delta
        
        return error

# Example usage
def main():
    # Create a neuron with 2 inputs (like an AND gate)
    neuron = Neuron(2)
    
    # Training data for AND gate
    training_data = [
        ([0, 0], 0),
        ([0, 1], 0),
        ([1, 0], 0),
        ([1, 1], 1)
    ]
    
    # Train the neuron
    epochs = 10000
    for epoch in range(epochs):
        total_error = 0
        for inputs, target in training_data:
            error = neuron.train(inputs, target)
            total_error += abs(error)
        
        # Print progress every 1000 epochs
        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch + 1}, Error: {total_error:.4f}")
    
    # Test the trained neuron
    print("\nTesting the trained neuron:")
    for inputs, _ in training_data:
        output = neuron.forward(inputs)
        print(f"Inputs: {inputs}, Output: {output:.4f}")

if __name__ == "__main__":
    main()
