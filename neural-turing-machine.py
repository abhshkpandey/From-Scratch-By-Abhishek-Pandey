import numpy as np
import math

class Head:
    def __init__(self, memory_size, memory_vector_dim):
        self.memory_size = memory_size
        self.memory_vector_dim = memory_vector_dim
        
        # Initialize parameters for attention mechanism
        self.k = np.zeros(memory_vector_dim)  # key vector
        self.beta = 0  # key strength
        self.g = 0  # gate for interpolation
        self.s = np.zeros(3)  # shift weighting
        self.gamma = 0  # sharpening factor
        
    def address_memory(self, memory, prev_weights):
        """
        Addressing mechanism using content and location-based addressing
        """
        # Content addressing
        w_c = self._content_addressing(memory)
        
        # Interpolation
        w_g = self._interpolate(w_c, prev_weights)
        
        # Shift
        w_s = self._shift(w_g)
        
        # Sharpen
        w = self._sharpen(w_s)
        
        return w
    
    def _content_addressing(self, memory):
        """
        Attention based on content similarity
        """
        # Calculate cosine similarity between key and each memory row
        similarities = np.zeros(self.memory_size)
        for i in range(self.memory_size):
            similarities[i] = self._cosine_similarity(self.k, memory[i])
        
        # Apply key strength (focus)
        similarities *= self.beta
        
        # Return softmax
        return self._softmax(similarities)
    
    def _interpolate(self, w_c, w_prev):
        """
        Interpolate between current and previous weights
        """
        return self.g * w_c + (1 - self.g) * w_prev
    
    def _shift(self, w):
        """
        Convolutional shift of weights
        """
        result = np.zeros(self.memory_size)
        for i in range(self.memory_size):
            for j in range(3):
                idx = (i - 1 + j) % self.memory_size
                result[i] += w[idx] * self.s[j]
        return result
    
    def _sharpen(self, w):
        """
        Sharpen the weight distribution
        """
        w = np.power(w, self.gamma)
        return w / np.sum(w)
    
    def _cosine_similarity(self, x, y):
        """
        Compute cosine similarity between two vectors
        """
        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)
        if norm_x == 0 or norm_y == 0:
            return 0
        return np.dot(x, y) / (norm_x * norm_y)
    
    def _softmax(self, x):
        """
        Compute softmax of vector
        """
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

class Controller:
    def __init__(self, input_size, output_size, memory_vector_dim, controller_size):
        self.input_size = input_size
        self.output_size = output_size
        self.memory_vector_dim = memory_vector_dim
        self.controller_size = controller_size
        
        # Initialize weights
        self.W_input = np.random.randn(controller_size, input_size) * 0.1
        self.W_read = np.random.randn(controller_size, memory_vector_dim) * 0.1
        self.W_output = np.random.randn(output_size, controller_size) * 0.1
        self.b_controller = np.zeros(controller_size)
        self.b_output = np.zeros(output_size)
        
        # Parameters for generating head parameters
        self.W_key = np.random.randn(memory_vector_dim, controller_size) * 0.1
        self.W_beta = np.random.randn(1, controller_size) * 0.1
        self.W_gate = np.random.randn(1, controller_size) * 0.1
        self.W_shift = np.random.randn(3, controller_size) * 0.1
        self.W_gamma = np.random.randn(1, controller_size) * 0.1
        
    def forward(self, x, prev_read):
        """
        Forward pass through the controller
        """
        # Concatenate input with previous read from memory
        combined_input = np.concatenate([x, prev_read])
        
        # Controller state
        h = np.tanh(np.dot(self.W_input, x) + np.dot(self.W_read, prev_read) + self.b_controller)
        
        # Generate head parameters
        key = np.tanh(np.dot(self.W_key, h))
        beta = np.exp(np.dot(self.W_beta, h))[0]
        gate = np.sigmoid(np.dot(self.W_gate, h))[0]
        shift = self._softmax(np.dot(self.W_shift, h))
        gamma = 1 + np.log(1 + np.exp(np.dot(self.W_gamma, h)))[0]
        
        # Generate output
        output = np.tanh(np.dot(self.W_output, h) + self.b_output)
        
        return output, (key, beta, gate, shift, gamma)
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

class NTM:
    def __init__(self, input_size, output_size, memory_size, memory_vector_dim, controller_size):
        self.memory_size = memory_size
        self.memory_vector_dim = memory_vector_dim
        
        # Initialize memory
        self.memory = np.zeros((memory_size, memory_vector_dim))
        
        # Initialize controller
        self.controller = Controller(
            input_size,
            output_size,
            memory_vector_dim,
            controller_size
        )
        
        # Initialize read and write heads
        self.read_head = Head(memory_size, memory_vector_dim)
        self.write_head = Head(memory_size, memory_vector_dim)
        
        # Initialize weights
        self.read_weights = np.zeros(memory_size)
        self.write_weights = np.zeros(memory_size)
        self.read_weights[0] = 1.0  # Focus on first location initially
        self.write_weights[0] = 1.0
        
        # Initialize previous read vector
        self.prev_read = np.zeros(memory_vector_dim)
    
    def forward(self, x):
        """
        Forward pass through the NTM
        """
        # Controller
        output, head_params = self.controller.forward(x, self.prev_read)
        
        # Update head parameters
        self._update_read_head(head_params)
        self._update_write_head(head_params)
        
        # Read from memory
        self.read_weights = self.read_head.address_memory(self.memory, self.read_weights)
        self.prev_read = np.dot(self.read_weights, self.memory)
        
        # Write to memory
        self.write_weights = self.write_head.address_memory(self.memory, self.write_weights)
        self._write_memory()
        
        return output
    
    def _update_read_head(self, head_params):
        key, beta, gate, shift, gamma = head_params
        self.read_head.k = key
        self.read_head.beta = beta
        self.read_head.g = gate
        self.read_head.s = shift
        self.read_head.gamma = gamma
    
    def _update_write_head(self, head_params):
        key, beta, gate, shift, gamma = head_params
        self.write_head.k = key
        self.write_head.beta = beta
        self.write_head.g = gate
        self.write_head.s = shift
        self.write_head.gamma = gamma
    
    def _write_memory(self):
        """
        Write to memory using attention mechanism
        """
        erase = np.outer(self.write_weights, np.ones(self.memory_vector_dim))
        add = np.outer(self.write_weights, self.controller.forward(np.zeros(self.controller.input_size), self.prev_read)[0])
        
        self.memory = self.memory * (1 - erase) + add

def main():
    # Example usage: Copy task
    # The NTM needs to memorize a sequence and reproduce it
    
    # Initialize NTM
    input_size = 10
    output_size = 10
    memory_size = 128
    memory_vector_dim = 20
    controller_size = 100
    
    ntm = NTM(input_size, output_size, memory_size, memory_vector_dim, controller_size)
    
    # Create example sequence
    sequence_length = 5
    sequence = np.random.randint(2, size=(sequence_length, input_size))
    
    print("Input sequence:")
    print(sequence)
    
    # Feed sequence through NTM
    outputs = []
    for x in sequence:
        output = ntm.forward(x)
        outputs.append(output)
    
    print("\nOutput sequence:")
    outputs = np.array(outputs)
    print(outputs)

if __name__ == "__main__":
    main()
