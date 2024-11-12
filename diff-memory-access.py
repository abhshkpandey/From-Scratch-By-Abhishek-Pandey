import numpy as np

class DifferentiableMemory:
    def __init__(self, memory_size, memory_vector_dim, controller_hidden_dim):
        """
        Initialize Differentiable Memory Access Module
        
        Args:
            memory_size: Number of memory slots
            memory_vector_dim: Dimension of each memory vector
            controller_hidden_dim: Hidden dimension of controller network
        """
        self.memory_size = memory_size
        self.memory_vector_dim = memory_vector_dim
        self.controller_hidden_dim = controller_hidden_dim
        
        # Initialize memory matrix
        self.memory = np.zeros((memory_size, memory_vector_dim))
        
        # Initialize controller weights
        self.W_controller = np.random.randn(controller_hidden_dim, memory_vector_dim) * 0.01
        self.b_controller = np.zeros((controller_hidden_dim, 1))
        
        # Initialize attention weights
        self.W_key = np.random.randn(memory_vector_dim, controller_hidden_dim) * 0.01
        self.b_key = np.zeros((memory_vector_dim, 1))
        
        # Initialize write weights
        self.W_write = np.random.randn(memory_vector_dim, controller_hidden_dim) * 0.01
        self.b_write = np.zeros((memory_vector_dim, 1))
        
        # Initialize erase weights
        self.W_erase = np.random.randn(memory_vector_dim, controller_hidden_dim) * 0.01
        self.b_erase = np.zeros((memory_vector_dim, 1))
        
        # Initialize interpolation gate weights
        self.W_interpolation = np.random.randn(1, controller_hidden_dim) * 0.01
        self.b_interpolation = np.zeros((1, 1))
        
        # Initialize shift weights
        self.shift_range = 1  # Shift range for convolutional shift
        self.W_shift = np.random.randn(2 * self.shift_range + 1, controller_hidden_dim) * 0.01
        self.b_shift = np.zeros((2 * self.shift_range + 1, 1))
        
        # Initialize sharpening weights
        self.W_gamma = np.random.randn(1, controller_hidden_dim) * 0.01
        self.b_gamma = np.ones((1, 1))  # Initialize bias to 1 for initial sharp focus

    def softmax(self, x, axis=-1):
        """Compute softmax values with numerical stability"""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def sigmoid(self, x):
        """Compute sigmoid activation"""
        return 1 / (1 + np.exp(-np.clip(x, -10, 10)))

    def cosine_similarity(self, key, memory):
        """
        Compute cosine similarity between key and memory
        
        Args:
            key: Query vector
            memory: Memory matrix
            
        Returns:
            similarity: Cosine similarity scores
        """
        key_norm = np.linalg.norm(key)
        memory_norm = np.linalg.norm(memory, axis=1)
        dot_product = np.dot(memory, key)
        similarity = dot_product / (key_norm * memory_norm + 1e-8)
        return similarity

    def circular_convolution(self, weights, shift_weights):
        """
        Apply circular convolution for shifting attention
        
        Args:
            weights: Current attention weights
            shift_weights: Learnable shift weights
            
        Returns:
            shifted_weights: Shifted attention weights
        """
        result = np.zeros_like(weights)
        for i in range(len(weights)):
            for j in range(-self.shift_range, self.shift_range + 1):
                idx = (i + j) % len(weights)
                result[i] += weights[idx] * shift_weights[j + self.shift_range]
        return result

    def read(self, controller_state, prev_weights=None):
        """
        Read from memory using differentiable attention
        
        Args:
            controller_state: Current controller state
            prev_weights: Previous attention weights (optional)
            
        Returns:
            read_vector: Read vector from memory
            weights: Attention weights used for reading
            cache: Values for backward pass
        """
        # Generate read key
        key = np.tanh(np.dot(self.W_key, controller_state) + self.b_key)
        
        # Compute content-based addressing
        similarity = self.cosine_similarity(key, self.memory)
        content_weights = self.softmax(similarity)
        
        if prev_weights is not None:
            # Interpolation gate
            interpolation = self.sigmoid(
                np.dot(self.W_interpolation, controller_state) + self.b_interpolation
            )
            
            # Interpolate between previous and content weights
            gated_weights = interpolation * content_weights + (1 - interpolation) * prev_weights
            
            # Convolutional shift
            shift_weights = self.softmax(
                np.dot(self.W_shift, controller_state) + self.b_shift
            )
            shifted_weights = self.circular_convolution(gated_weights, shift_weights)
            
            # Sharpen weights
            gamma = 1 + self.sigmoid(
                np.dot(self.W_gamma, controller_state) + self.b_gamma
            )
            weights = np.power(shifted_weights, gamma)
            weights = weights / np.sum(weights)
        else:
            weights = content_weights
        
        # Read from memory
        read_vector = np.dot(weights, self.memory)
        
        cache = (key, similarity, content_weights, weights)
        return read_vector, weights, cache

    def write(self, controller_state, weights):
        """
        Write to memory using differentiable attention
        
        Args:
            controller_state: Current controller state
            weights: Attention weights for writing
            
        Returns:
            cache: Values for backward pass
        """
        # Generate write and erase vectors
        write_vector = np.tanh(np.dot(self.W_write, controller_state) + self.b_write)
        erase_vector = self.sigmoid(np.dot(self.W_erase, controller_state) + self.b_erase)
        
        # Erase operation
        erase_matrix = np.outer(weights, erase_vector)
        self.memory = self.memory * (1 - erase_matrix)
        
        # Write operation
        write_matrix = np.outer(weights, write_vector)
        self.memory = self.memory + write_matrix
        
        cache = (write_vector, erase_vector, erase_matrix, write_matrix)
        return cache

class DifferentiableMemoryController:
    def __init__(self, input_size, memory_size, memory_vector_dim, controller_hidden_dim):
        """
        Initialize controller for Differentiable Memory Access
        
        Args:
            input_size: Size of input vector
            memory_size: Number of memory slots
            memory_vector_dim: Dimension of each memory vector
            controller_hidden_dim: Hidden dimension of controller network
        """
        self.memory = DifferentiableMemory(memory_size, memory_vector_dim, controller_hidden_dim)
        
        # Initialize controller weights
        self.W_input = np.random.randn(controller_hidden_dim, input_size) * 0.01
        self.W_read = np.random.randn(controller_hidden_dim, memory_vector_dim) * 0.01
        self.b_controller = np.zeros((controller_hidden_dim, 1))
        
        # Initialize output weights
        self.W_output = np.random.randn(memory_vector_dim, controller_hidden_dim) * 0.01
        self.b_output = np.zeros((memory_vector_dim, 1))

    def forward(self, x, prev_state=None, prev_weights=None):
        """
        Forward pass through controller and memory access
        
        Args:
            x: Input vector
            prev_state: Previous controller state (optional)
            prev_weights: Previous attention weights (optional)
            
        Returns:
            output: Network output
            state: New controller state
            weights: New attention weights
            cache: Values for backward pass
        """
        # Controller network
        if prev_state is None:
            state = np.tanh(np.dot(self.W_input, x) + self.b_controller)
        else:
            state = np.tanh(
                np.dot(self.W_input, x) + 
                np.dot(self.W_read, prev_state) + 
                self.b_controller
            )
        
        # Read from memory
        read_vector, weights, read_cache = self.memory.read(state, prev_weights)
        
        # Write to memory
        write_cache = self.memory.write(state, weights)
        
        # Generate output
        output = np.dot(self.W_output, state) + self.b_output
        
        cache = (x, state, read_vector, read_cache, write_cache)
        return output, state, weights, cache

def example_usage():
    # Initialize parameters
    input_size = 10
    memory_size = 128
    memory_vector_dim = 20
    controller_hidden_dim = 32
    
    # Create controller
    controller = DifferentiableMemoryController(
        input_size, memory_size, memory_vector_dim, controller_hidden_dim
    )
    
    # Create sample input
    x = np.random.randn(input_size, 1)
    
    # Forward pass
    output, state, weights, cache = controller.forward(x)
    
    print("Output shape:", output.shape)
    print("Controller state shape:", state.shape)
    print("Attention weights shape:", weights.shape)
    print("Memory shape:", controller.memory.memory.shape)

if __name__ == "__main__":
    example_usage()
