import numpy as np

class MANN:
    def __init__(self, input_size, output_size, memory_size, memory_vector_dim, controller_hidden_dim):
        """
        Initialize Memory-Augmented Neural Network
        
        Args:
            input_size: Size of input vector
            output_size: Size of output vector
            memory_size: Number of memory slots
            memory_vector_dim: Dimension of each memory vector
            controller_hidden_dim: Hidden dimension of controller network
        """
        self.input_size = input_size
        self.output_size = output_size
        self.memory_size = memory_size
        self.memory_vector_dim = memory_vector_dim
        self.controller_hidden_dim = controller_hidden_dim
        
        # Initialize controller network weights
        self.W_controller = np.random.randn(controller_hidden_dim, input_size) * 0.01
        self.U_controller = np.random.randn(controller_hidden_dim, controller_hidden_dim) * 0.01
        self.b_controller = np.zeros((controller_hidden_dim, 1))
        
        # Initialize output weights
        self.W_output = np.random.randn(output_size, controller_hidden_dim) * 0.01
        self.b_output = np.zeros((output_size, 1))
        
        # Initialize memory addressing weights
        self.W_key = np.random.randn(memory_vector_dim, controller_hidden_dim) * 0.01
        self.b_key = np.zeros((memory_vector_dim, 1))
        
        # Initialize memory
        self.memory = np.zeros((memory_size, memory_vector_dim))
        
        # Initialize write weights
        self.W_write = np.random.randn(memory_vector_dim, controller_hidden_dim) * 0.01
        self.b_write = np.zeros((memory_vector_dim, 1))
        
        # Initialize erase vector weights
        self.W_erase = np.random.randn(memory_vector_dim, controller_hidden_dim) * 0.01
        self.b_erase = np.zeros((memory_vector_dim, 1))

    def cosine_similarity(self, key, memory):
        """Calculate cosine similarity between a key and all memory locations"""
        dot_product = np.dot(memory, key)
        key_norm = np.linalg.norm(key)
        memory_norm = np.linalg.norm(memory, axis=1)
        similarity = dot_product / (key_norm * memory_norm + 1e-8)
        return similarity

    def get_addressing_weights(self, key, memory):
        """Get softmax addressing weights using cosine similarity"""
        similarity = self.cosine_similarity(key, memory)
        weights = np.exp(similarity) / (np.sum(np.exp(similarity)) + 1e-8)
        return weights

    def controller_network(self, x, prev_state):
        """Feed-forward controller network with previous state"""
        h = np.tanh(np.dot(self.W_controller, x) + np.dot(self.U_controller, prev_state) + self.b_controller)
        return h

    def read_memory(self, weights):
        """Read from memory using addressing weights"""
        return np.dot(weights, self.memory)

    def write_memory(self, weights, write_vector, erase_vector):
        """Write to memory using addressing weights"""
        # Erase operation
        erase_term = np.outer(weights, erase_vector)
        self.memory = self.memory * (1 - erase_term)
        
        # Write operation
        write_term = np.outer(weights, write_vector)
        self.memory = self.memory + write_term

    def forward(self, x, prev_state):
        """
        Forward pass of MANN
        
        Args:
            x: Input vector
            prev_state: Previous controller state
            
        Returns:
            output: Network output
            state: New controller state
            read_vector: Read vector from memory
            cache: Values needed for backward pass
        """
        # Controller network
        state = self.controller_network(x, prev_state)
        
        # Generate read key
        key = np.tanh(np.dot(self.W_key, state) + self.b_key)
        
        # Get addressing weights
        weights = self.get_addressing_weights(key, self.memory)
        
        # Read from memory
        read_vector = self.read_memory(weights)
        
        # Generate write vector and erase vector
        write_vector = np.tanh(np.dot(self.W_write, state) + self.b_write)
        erase_vector = self.sigmoid(np.dot(self.W_erase, state) + self.b_erase)
        
        # Write to memory
        self.write_memory(weights, write_vector, erase_vector)
        
        # Generate output
        output = np.dot(self.W_output, state) + self.b_output
        
        cache = (x, prev_state, state, key, weights, write_vector, erase_vector, read_vector)
        return output, state, read_vector, cache

    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))

# Example usage
def example_usage():
    # Initialize parameters
    input_size = 10
    output_size = 5
    memory_size = 128
    memory_vector_dim = 20
    controller_hidden_dim = 32
    
    # Create MANN instance
    mann = MANN(input_size, output_size, memory_size, memory_vector_dim, controller_hidden_dim)
    
    # Create sample input
    x = np.random.randn(input_size, 1)
    prev_state = np.zeros((controller_hidden_dim, 1))
    
    # Forward pass
    output, new_state, read_vector, cache = mann.forward(x, prev_state)
    
    print("Output shape:", output.shape)
    print("Controller state shape:", new_state.shape)
    print("Read vector shape:", read_vector.shape)
    print("Memory shape:", mann.memory.shape)

    return mann

# Test the implementation
if __name__ == "__main__":
    mann = example_usage()
