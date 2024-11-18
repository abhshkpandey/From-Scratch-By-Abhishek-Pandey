import numpy as np

class NTMController:
    def __init__(self, input_size, output_size, controller_size, memory_size, memory_vector_dim):
        """
        Initialize NTM Controller
        
        Args:
            input_size: Size of input vector
            output_size: Size of output vector
            controller_size: Size of controller hidden layer
            memory_size: Number of memory locations
            memory_vector_dim: Size of each memory vector
        """
        self.input_size = input_size
        self.output_size = output_size
        self.controller_size = controller_size
        self.memory_size = memory_size
        self.memory_vector_dim = memory_vector_dim
        
        # Initialize controller weights
        self.W_controller = np.random.randn(controller_size, input_size + memory_vector_dim) * 0.01
        self.b_controller = np.zeros((controller_size, 1))
        
        # Initialize output weights
        self.W_output = np.random.randn(output_size, controller_size) * 0.01
        self.b_output = np.zeros((output_size, 1))
        
        # Initialize memory interface weights
        self.W_key = np.random.randn(memory_vector_dim, controller_size) * 0.01
        self.W_beta = np.random.randn(1, controller_size) * 0.01
        self.W_g = np.random.randn(1, controller_size) * 0.01
        self.W_shift = np.random.randn(3, controller_size) * 0.01  # 3 shift operations: -1, 0, 1
        self.W_gamma = np.random.randn(1, controller_size) * 0.01
        self.W_erase = np.random.randn(memory_vector_dim, controller_size) * 0.01
        self.W_write = np.random.randn(memory_vector_dim, controller_size) * 0.01
        
        # Initialize memory
        self.memory = np.zeros((memory_size, memory_vector_dim))
        
        # Previous read weights
        self.prev_read_weights = np.zeros((memory_size, 1))
        
        # Previous write weights
        self.prev_write_weights = np.zeros((memory_size, 1))

    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -10, 10)))

    def softmax(self, x, axis=-1):
        """Softmax with numerical stability"""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def cosine_similarity(self, key, memory):
        """Compute cosine similarity between key and memory"""
        key_norm = np.linalg.norm(key)
        memory_norm = np.linalg.norm(memory, axis=1, keepdims=True)
        dot_product = np.dot(memory, key)
        return dot_product / (key_norm * memory_norm + 1e-8)

    def get_addressing_weights(self, key, beta, g, shift, gamma, prev_weights):
        """
        Generate addressing weights using content and location-based addressing
        
        Args:
            key: Key vector for content-based addressing
            beta: Key strength
            g: Interpolation gate
            shift: Shift weighting
            gamma: Sharpening factor
            prev_weights: Previous weights
        """
        # Content addressing
        similarity = self.cosine_similarity(key, self.memory)
        content_weights = self.softmax(beta * similarity)
        
        # Interpolation
        interpolated_weights = g * content_weights + (1 - g) * prev_weights
        
        # Shifting
        shifted_weights = np.zeros_like(interpolated_weights)
        for i in range(self.memory_size):
            for j in range(3):
                idx = (i - 1 + j) % self.memory_size
                shifted_weights[i] += interpolated_weights[idx] * shift[j]
        
        # Sharpening
        sharp_weights = np.power(shifted_weights, gamma)
        return sharp_weights / (np.sum(sharp_weights) + 1e-8)

    def read(self, weights):
        """Read from memory using attention weights"""
        return np.dot(weights.T, self.memory)

    def write(self, weights, erase_vector, write_vector):
        """Write to memory using attention weights"""
        # Erase
        erase_matrix = np.outer(weights, erase_vector)
        self.memory *= (1 - erase_matrix)
        
        # Write
        write_matrix = np.outer(weights, write_vector)
        self.memory += write_matrix

    def forward(self, x):
        """
        Forward pass of NTM
        
        Args:
            x: Input vector
            
        Returns:
            output: Network output
            read_weights: Read attention weights
            write_weights: Write attention weights
            cache: Values for backward pass
        """
        # Read from memory
        prev_read = self.read(self.prev_read_weights)
        
        # Concatenate input with previous read
        controller_input = np.vstack([x, prev_read.T])
        
        # Controller network
        controller_state = np.tanh(np.dot(self.W_controller, controller_input) + self.b_controller)
        
        # Generate interface parameters
        key = np.tanh(np.dot(self.W_key, controller_state))
        beta = self.sigmoid(np.dot(self.W_beta, controller_state))
        g = self.sigmoid(np.dot(self.W_g, controller_state))
        shift = self.softmax(np.dot(self.W_shift, controller_state))
        gamma = 1 + self.sigmoid(np.dot(self.W_gamma, controller_state))
        erase_vector = self.sigmoid(np.dot(self.W_erase, controller_state))
        write_vector = np.tanh(np.dot(self.W_write, controller_state))
        
        # Get read and write weights
        read_weights = self.get_addressing_weights(
            key, beta, g, shift, gamma, self.prev_read_weights
        )
        write_weights = self.get_addressing_weights(
            key, beta, g, shift, gamma, self.prev_write_weights
        )
        
        # Read from memory
        read_vector = self.read(read_weights)
        
        # Write to memory
        self.write(write_weights, erase_vector, write_vector)
        
        # Generate output
        output = self.sigmoid(np.dot(self.W_output, controller_state) + self.b_output)
        
        # Update previous weights
        self.prev_read_weights = read_weights
        self.prev_write_weights = write_weights
        
        cache = (controller_input, controller_state, key, beta, g, shift, gamma,
                erase_vector, write_vector, read_weights, write_weights)
        return output, read_weights, write_weights, cache

class SequenceNTM:
    def __init__(self, sequence_width, controller_size=128, memory_size=128, memory_vector_dim=20):
        """
        Initialize Sequence NTM
        
        Args:
            sequence_width: Width of input/output sequences
            controller_size: Size of controller hidden layer
            memory_size: Number of memory locations
            memory_vector_dim: Size of each memory vector
        """
        self.ntm = NTMController(
            input_size=sequence_width + 2,  # +2 for start and end flags
            output_size=sequence_width,
            controller_size=controller_size,
            memory_size=memory_size,
            memory_vector_dim=memory_vector_dim
        )
        self.sequence_width = sequence_width

    def train_sequence(self, sequence, max_length=20):
        """
        Train NTM on a sequence copying task
        
        Args:
            sequence: Input sequence (binary vectors)
            max_length: Maximum sequence length
        """
        # Add start flag
        x = np.vstack([1, np.zeros(self.sequence_width), sequence])
        
        # Forward pass through sequence
        outputs = []
        read_weights_history = []
        write_weights_history = []
        
        for t in range(len(x)):
            output, read_weights, write_weights, _ = self.ntm.forward(x[t:t+1].T)
            outputs.append(output)
            read_weights_history.append(read_weights)
            write_weights_history.append(write_weights)
        
        # Add end flag and retrieve sequence
        x_end = np.vstack([0, 1, np.zeros(self.sequence_width)])
        retrieved_sequence = []
        
        for _ in range(len(sequence)):
            output, read_weights, write_weights, _ = self.ntm.forward(x_end)
            retrieved_sequence.append(output)
            read_weights_history.append(read_weights)
            write_weights_history.append(write_weights)
        
        return np.array(retrieved_sequence), np.array(read_weights_history), np.array(write_weights_history)

def example_usage():
    # Parameters
    sequence_width = 8
    sequence_length = 5
    
    # Create random binary sequence
    sequence = np.random.randint(0, 2, (sequence_length, sequence_width))
    
    # Initialize Sequence NTM
    ntm = SequenceNTM(sequence_width)
    
    # Train on sequence
    retrieved_sequence, read_weights, write_weights = ntm.train_sequence(sequence)
    
    print("Original sequence shape:", sequence.shape)
    print("Retrieved sequence shape:", retrieved_sequence.shape)
    print("Read weights history shape:", read_weights.shape)
    print("Write weights history shape:", write_weights.shape)
    
    # Compare original and retrieved sequences
    print("\nOriginal sequence:")
    print(sequence)
    print("\nRetrieved sequence:")
    print(np.round(retrieved_sequence))
    
    return ntm, sequence, retrieved_sequence, read_weights, write_weights

if __name__ == "__main__":
    ntm, sequence, retrieved_sequence, read_weights, write_weights = example_usage()
