import numpy as np
import math
import random

class RNNController:
    def __init__(self, input_size, memory_vector_dim, hidden_size, output_size):
        # RNN parameters
        self.hidden_size = hidden_size
        self.memory_vector_dim = memory_vector_dim
        
        # Initialize weights for the RNN part
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Wrh = np.random.randn(hidden_size, memory_vector_dim) * 0.01  # For memory read input
        self.bh = np.zeros((hidden_size, 1))
        
        # Output weights
        self.Why = np.random.randn(output_size, hidden_size) * 0.01
        self.by = np.zeros((output_size, 1))
        
        # Memory interface weights
        self.Whk = np.random.randn(memory_vector_dim, hidden_size) * 0.01  # Key generation
        self.Whβ = np.random.randn(1, hidden_size) * 0.01  # Key strength
        self.Whg = np.random.randn(1, hidden_size) * 0.01  # Gate
        self.Whs = np.random.randn(3, hidden_size) * 0.01  # Shift
        self.Whγ = np.random.randn(1, hidden_size) * 0.01  # Sharpening
        self.Whe = np.random.randn(memory_vector_dim, hidden_size) * 0.01  # Erase vector
        self.Wha = np.random.randn(memory_vector_dim, hidden_size) * 0.01  # Add vector
        
        # Initialize states
        self.hidden_state = np.zeros((hidden_size, 1))
        
    def forward(self, x, prev_read):
        """
        Forward pass through the RNN controller
        x: input vector
        prev_read: previous read vector from memory
        """
        # Convert inputs to column vectors
        x = np.array(x).reshape(-1, 1)
        prev_read = np.array(prev_read).reshape(-1, 1)
        
        # RNN forward pass
        self.hidden_state = np.tanh(
            np.dot(self.Wxh, x) +
            np.dot(self.Whh, self.hidden_state) +
            np.dot(self.Wrh, prev_read) +
            self.bh
        )
        
        # Generate output
        output = np.tanh(np.dot(self.Why, self.hidden_state) + self.by)
        
        # Generate memory control signals
        key = np.tanh(np.dot(self.Whk, self.hidden_state))
        beta = np.exp(np.dot(self.Whβ, self.hidden_state))
        gate = self.sigmoid(np.dot(self.Whg, self.hidden_state))
        shift = self.softmax(np.dot(self.Whs, self.hidden_state))
        gamma = 1 + np.log(1 + np.exp(np.dot(self.Whγ, self.hidden_state)))
        erase = self.sigmoid(np.dot(self.Whe, self.hidden_state))
        add = np.tanh(np.dot(self.Wha, self.hidden_state))
        
        return output, (key, beta, gate, shift, gamma, erase, add)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

class MemoryHead:
    def __init__(self, memory_size, memory_vector_dim):
        self.memory_size = memory_size
        self.memory_vector_dim = memory_vector_dim
        self.prev_weights = np.zeros(memory_size)
        self.prev_weights[0] = 1.0  # Initialize focus to first position
    
    def address_memory(self, memory, key, beta, gate, shift, gamma):
        """
        Generate attention weights over memory locations
        """
        # Content addressing
        content_weights = self._content_addressing(memory, key, beta)
        
        # Interpolation
        gated_weights = gate * content_weights + (1 - gate) * self.prev_weights
        
        # Convolutional shift
        shifted_weights = self._shift_weights(gated_weights, shift)
        
        # Sharpening
        sharp_weights = self._sharpen(shifted_weights, gamma)
        
        self.prev_weights = sharp_weights
        return sharp_weights
    
    def _content_addressing(self, memory, key, beta):
        """
        Compute content-based addressing weights
        """
        # Compute cosine similarity
        similarities = np.zeros(self.memory_size)
        for i in range(self.memory_size):
            similarities[i] = self._cosine_similarity(key.flatten(), memory[i])
        
        # Apply key strength
        similarities = similarities * beta
        
        # Return normalized weights
        return self.softmax(similarities)
    
    def _shift_weights(self, weights, shift):
        """
        Circular convolution of weights with shift vector
        """
        result = np.zeros(self.memory_size)
        for i in range(self.memory_size):
            for j in range(len(shift)):
                idx = (i - j + 1) % self.memory_size
                result[i] += weights[idx] * shift[j]
        return result
    
    def _sharpen(self, weights, gamma):
        """
        Sharpen the focus of attention weights
        """
        weights = np.power(weights, gamma)
        return weights / np.sum(weights)
    
    def _cosine_similarity(self, x, y):
        """
        Compute cosine similarity between two vectors
        """
        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)
        if norm_x == 0 or norm_y == 0:
            return 0
        return np.dot(x, y) / (norm_x * norm_y)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

class RNNNTM:
    def __init__(self, input_size, memory_size, memory_vector_dim, hidden_size, output_size):
        """
        Initialize RNN-based Neural Turing Machine
        """
        self.memory_size = memory_size
        self.memory_vector_dim = memory_vector_dim
        
        # Initialize memory
        self.memory = np.zeros((memory_size, memory_vector_dim))
        
        # Initialize controller
        self.controller = RNNController(
            input_size,
            memory_vector_dim,
            hidden_size,
            output_size
        )
        
        # Initialize heads
        self.read_head = MemoryHead(memory_size, memory_vector_dim)
        self.write_head = MemoryHead(memory_size, memory_vector_dim)
        
        # Initialize read vector
        self.prev_read = np.zeros(memory_vector_dim)
    
    def forward(self, x):
        """
        Forward pass through the RNN-NTM
        """
        # Controller
        output, params = self.controller.forward(x, self.prev_read)
        key, beta, gate, shift, gamma, erase, add = params
        
        # Read Operation
        read_weights = self.read_head.address_memory(
            self.memory,
            key,
            beta,
            gate,
            shift,
            gamma
        )
        
        self.prev_read = np.dot(read_weights, self.memory)
        
        # Write Operation
        write_weights = self.write_head.address_memory(
            self.memory,
            key,
            beta,
            gate,
            shift,
            gamma
        )
        
        # Erase and Add
        erase_matrix = np.outer(write_weights, erase.flatten())
        add_matrix = np.outer(write_weights, add.flatten())
        
        self.memory = self.memory * (1 - erase_matrix) + add_matrix
        
        return output.flatten()
    
    def reset_state(self):
        """
        Reset the state of the network
        """
        self.controller.hidden_state = np.zeros_like(self.controller.hidden_state)
        self.prev_read = np.zeros(self.memory_vector_dim)
        self.read_head.prev_weights = np.zeros(self.memory_size)
        self.write_head.prev_weights = np.zeros(self.memory_size)
        self.read_head.prev_weights[0] = 1.0
        self.write_head.prev_weights[0] = 1.0
        self.memory = np.zeros((self.memory_size, self.memory_vector_dim))

def train_copy_task(ntm, sequence_length, num_epochs=1000):
    """
    Train the RNN-NTM on a sequence copying task
    """
    for epoch in range(num_epochs):
        # Generate random sequence
        sequence = np.random.randint(2, size=(sequence_length, 8))
        target_sequence = sequence.copy()
        
        # Reset NTM state
        ntm.reset_state()
        
        total_loss = 0
        outputs = []
        
        # Present input sequence
        for input_vector in sequence:
            output = ntm.forward(np.append(input_vector, [1, 0]))  # Add flags
            outputs.append(output[:-2])  # Remove flags from output
        
        # Present delimiter
        output = ntm.forward(np.array([0] * 8 + [0, 1]))
        outputs.append(output[:-2])
        
        # Read output sequence
        for _ in range(sequence_length):
            output = ntm.forward(np.array([0] * 10))
            outputs.append(output[:-2])
        
        # Compute loss (only on the output phase)
        outputs = np.array(outputs)
        loss = np.mean((outputs[-sequence_length:] - target_sequence) ** 2)
        total_loss += loss
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

def main():
    # Initialize RNN-NTM
    input_size = 10  # 8 bits + 2 flags
    memory_size = 128
    memory_vector_dim = 20
    hidden_size = 100
    output_size = 10  # 8 bits + 2 flags
    
    rnn_ntm = RNNNTM(
        input_size,
        memory_size,
        memory_vector_dim,
        hidden_size,
        output_size
    )
    
    # Example: Copy Task
    print("Training RNN-NTM on copy task...")
    sequence_length = 5
    train_copy_task(rnn_ntm, sequence_length)
    
    # Test the trained network
    print("\nTesting RNN-NTM...")
    test_sequence = np.random.randint(2, size=(sequence_length, 8))
    print("Input sequence:")
    print(test_sequence)
    
    rnn_ntm.reset_state()
    outputs = []
    
    # Present input sequence
    for input_vector in test_sequence:
        output = rnn_ntm.forward(np.append(input_vector, [1, 0]))
        outputs.append(output[:-2])
    
    # Present delimiter
    output = rnn_ntm.forward(np.array([0] * 8 + [0, 1]))
    outputs.append(output[:-2])
    
    # Read output sequence
    for _ in range(sequence_length):
        output = rnn_ntm.forward(np.array([0] * 10))
        outputs.append(output[:-2])
    
    print("\nOutput sequence:")
    outputs = np.array(outputs)
    print(outputs[-sequence_length:])

if __name__ == "__main__":
    main()
