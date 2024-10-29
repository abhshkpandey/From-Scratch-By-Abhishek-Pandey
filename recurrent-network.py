import random
import math
import copy

class RNNNeuron:
    def __init__(self, num_inputs, num_recurrent):
        # Initialize weights for regular inputs
        self.input_weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
        # Initialize weights for recurrent connections
        self.recurrent_weights = [random.uniform(-1, 1) for _ in range(num_recurrent)]
        self.bias = random.uniform(-1, 1)
        
        # Store states and gradients for backpropagation through time (BPTT)
        self.states = []
        self.outputs = []
    
    def tanh(self, x):
        # Using tanh as activation function (better for RNNs than sigmoid)
        return math.tanh(x)
    
    def tanh_derivative(self, x):
        # Derivative of tanh
        return 1.0 - math.tanh(x)**2
    
    def forward(self, inputs, prev_outputs):
        # Combine current inputs with previous outputs using respective weights
        weighted_sum_inputs = sum(w * x for w, x in zip(self.input_weights, inputs))
        weighted_sum_recurrent = sum(w * x for w, x in zip(self.recurrent_weights, prev_outputs))
        
        # Calculate total input
        self.weighted_sum = weighted_sum_inputs + weighted_sum_recurrent + self.bias
        
        # Apply activation function
        output = self.tanh(self.weighted_sum)
        
        # Store states for backpropagation
        self.states.append((inputs, prev_outputs, self.weighted_sum))
        self.outputs.append(output)
        
        return output

class RNNLayer:
    def __init__(self, num_neurons, num_inputs, num_recurrent):
        self.neurons = [RNNNeuron(num_inputs, num_recurrent) for _ in range(num_neurons)]
        self.prev_outputs = [0.0] * num_neurons
    
    def forward(self, inputs):
        current_outputs = []
        for neuron in self.neurons:
            output = neuron.forward(inputs, self.prev_outputs)
            current_outputs.append(output)
        
        self.prev_outputs = current_outputs
        return current_outputs
    
    def reset_state(self):
        self.prev_outputs = [0.0] * len(self.neurons)
        for neuron in self.neurons:
            neuron.states = []
            neuron.outputs = []

class RNN:
    def __init__(self, input_size, hidden_size, output_size):
        """
        input_size: number of input features
        hidden_size: number of neurons in the hidden layer
        output_size: number of output features
        """
        # Create hidden RNN layer
        self.hidden_layer = RNNLayer(hidden_size, input_size, hidden_size)
        # Create output layer (regular feedforward layer)
        self.output_layer = RNNLayer(output_size, hidden_size, 0)
    
    def forward(self, sequence):
        """
        Forward pass through the network for a sequence of inputs
        sequence: list of input vectors
        """
        hidden_outputs = []
        final_outputs = []
        
        for inputs in sequence:
            # Process through hidden layer
            hidden = self.hidden_layer.forward(inputs)
            hidden_outputs.append(hidden)
            
            # Process through output layer
            output = self.output_layer.forward(hidden)
            final_outputs.append(output)
        
        return final_outputs
    
    def backpropagation_through_time(self, sequence, targets, learning_rate=0.01):
        """
        Implement truncated backpropagation through time (BPTT)
        """
        sequence_length = len(sequence)
        
        # Initialize gradients
        hidden_size = len(self.hidden_layer.neurons)
        input_size = len(sequence[0])
        
        for t in range(sequence_length - 1, -1, -1):
            # Calculate output layer error
            output_deltas = []
            for i, neuron in enumerate(self.output_layer.neurons):
                error = targets[t][i] - neuron.outputs[t]
                delta = error * neuron.tanh_derivative(neuron.states[t][2])
                output_deltas.append(delta)
            
            # Calculate hidden layer error
            hidden_deltas = []
            for i, neuron in enumerate(self.hidden_layer.neurons):
                error = 0
                # Error from output layer
                for j, output_neuron in enumerate(self.output_layer.neurons):
                    error += output_neuron.input_weights[i] * output_deltas[j]
                
                # Error from next time step (if not last in sequence)
                if t < sequence_length - 1:
                    for j, hidden_neuron in enumerate(self.hidden_layer.neurons):
                        error += hidden_neuron.recurrent_weights[i] * hidden_deltas_prev[j]
                
                delta = error * neuron.tanh_derivative(neuron.states[t][2])
                hidden_deltas.append(delta)
            
            # Store hidden deltas for next iteration
            hidden_deltas_prev = hidden_deltas
            
            # Update weights
            # Output layer
            for i, neuron in enumerate(self.output_layer.neurons):
                hidden_outputs = neuron.states[t][0]  # Hidden layer outputs
                for j in range(len(neuron.input_weights)):
                    neuron.input_weights[j] += learning_rate * output_deltas[i] * hidden_outputs[j]
                neuron.bias += learning_rate * output_deltas[i]
            
            # Hidden layer
            for i, neuron in enumerate(self.hidden_layer.neurons):
                inputs = neuron.states[t][0]  # Current inputs
                prev_hidden = neuron.states[t][1]  # Previous hidden states
                
                # Update input weights
                for j in range(len(neuron.input_weights)):
                    neuron.input_weights[j] += learning_rate * hidden_deltas[i] * inputs[j]
                
                # Update recurrent weights
                for j in range(len(neuron.recurrent_weights)):
                    neuron.recurrent_weights[j] += learning_rate * hidden_deltas[i] * prev_hidden[j]
                
                neuron.bias += learning_rate * hidden_deltas[i]
    
    def train(self, sequences, targets, epochs, learning_rate=0.01):
        """
        Train the RNN on multiple sequences
        sequences: list of input sequences
        targets: list of target sequences
        """
        for epoch in range(epochs):
            total_error = 0
            
            for sequence, target in zip(sequences, targets):
                # Reset states at the start of each sequence
                self.hidden_layer.reset_state()
                self.output_layer.reset_state()
                
                # Forward pass
                outputs = self.forward(sequence)
                
                # Calculate error
                for t in range(len(sequence)):
                    for o, t in zip(outputs[t], target[t]):
                        total_error += (t - o) ** 2
                
                # Backward pass
                self.backpropagation_through_time(sequence, target, learning_rate)
            
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}, Error: {total_error:.4f}")
    
    def predict(self, sequence):
        """Make a prediction for a single sequence"""
        self.hidden_layer.reset_state()
        self.output_layer.reset_state()
        return self.forward(sequence)

def main():
    # Example: Learning a simple sequence pattern
    # In this example, we'll train the RNN to output the next number in a sequence
    
    # Create RNN with 1 input feature, 4 hidden neurons, and 1 output
    rnn = RNN(input_size=1, hidden_size=4, output_size=1)
    
    # Create training data
    # Example: Given a number, predict the next number in the sequence [0.1, 0.2, 0.3, 0.4, 0.5]
    sequences = [
        [[0.1], [0.2], [0.3], [0.4]],
        [[0.2], [0.3], [0.4], [0.5]],
    ]
    
    targets = [
        [[0.2], [0.3], [0.4], [0.5]],
        [[0.3], [0.4], [0.5], [0.6]],
    ]
    
    # Train the network
    print("Training network...")
    rnn.train(sequences, targets, epochs=1000, learning_rate=0.01)
    
    # Test the network
    print("\nTesting network:")
    test_sequence = [[0.1], [0.2], [0.3], [0.4]]
    predictions = rnn.predict(test_sequence)
    
    print("\nPredictions for sequence [0.1, 0.2, 0.3, 0.4]:")
    for t, pred in enumerate(predictions):
        print(f"Input: {test_sequence[t][0]:.1f}, Predicted next: {pred[0]:.4f}")

if __name__ == "__main__":
    main()
