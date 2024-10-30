import numpy as np

class LSTM:
    def __init__(self, input_size, hidden_size):
        """
        Initialize LSTM cell
        
        Args:
            input_size: Size of input vector
            hidden_size: Size of hidden state vector
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Initialize weights and biases
        # Input gate
        self.Wi = np.random.randn(hidden_size, input_size) * 0.01
        self.Ui = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bi = np.zeros((hidden_size, 1))
        
        # Forget gate
        self.Wf = np.random.randn(hidden_size, input_size) * 0.01
        self.Uf = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bf = np.zeros((hidden_size, 1))
        
        # Output gate
        self.Wo = np.random.randn(hidden_size, input_size) * 0.01
        self.Uo = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bo = np.zeros((hidden_size, 1))
        
        # Cell state
        self.Wc = np.random.randn(hidden_size, input_size) * 0.01
        self.Uc = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bc = np.zeros((hidden_size, 1))

    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        """Tanh activation function"""
        return np.tanh(x)

    def forward(self, x, prev_h, prev_c):
        """
        Forward pass of LSTM
        
        Args:
            x: Input vector
            prev_h: Previous hidden state
            prev_c: Previous cell state
            
        Returns:
            h: New hidden state
            c: New cell state
            cache: Values needed for backward pass
        """
        # Input gate
        i = self.sigmoid(np.dot(self.Wi, x) + np.dot(self.Ui, prev_h) + self.bi)
        
        # Forget gate
        f = self.sigmoid(np.dot(self.Wf, x) + np.dot(self.Uf, prev_h) + self.bf)
        
        # Output gate
        o = self.sigmoid(np.dot(self.Wo, x) + np.dot(self.Uo, prev_h) + self.bo)
        
        # Cell state candidate
        c_hat = self.tanh(np.dot(self.Wc, x) + np.dot(self.Uc, prev_h) + self.bc)
        
        # New cell state
        c = f * prev_c + i * c_hat
        
        # New hidden state
        h = o * self.tanh(c)
        
        cache = (x, prev_h, prev_c, i, f, o, c_hat, c)
        return h, c, cache

class GRU:
    def __init__(self, input_size, hidden_size):
        """
        Initialize GRU cell
        
        Args:
            input_size: Size of input vector
            hidden_size: Size of hidden state vector
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Update gate
        self.Wz = np.random.randn(hidden_size, input_size) * 0.01
        self.Uz = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bz = np.zeros((hidden_size, 1))
        
        # Reset gate
        self.Wr = np.random.randn(hidden_size, input_size) * 0.01
        self.Ur = np.random.randn(hidden_size, hidden_size) * 0.01
        self.br = np.zeros((hidden_size, 1))
        
        # Candidate hidden state
        self.Wh = np.random.randn(hidden_size, input_size) * 0.01
        self.Uh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))

    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        """Tanh activation function"""
        return np.tanh(x)

    def forward(self, x, prev_h):
        """
        Forward pass of GRU
        
        Args:
            x: Input vector
            prev_h: Previous hidden state
            
        Returns:
            h: New hidden state
            cache: Values needed for backward pass
        """
        # Update gate
        z = self.sigmoid(np.dot(self.Wz, x) + np.dot(self.Uz, prev_h) + self.bz)
        
        # Reset gate
        r = self.sigmoid(np.dot(self.Wr, x) + np.dot(self.Ur, prev_h) + self.br)
        
        # Candidate hidden state
        h_hat = self.tanh(np.dot(self.Wh, x) + np.dot(self.Uh, (r * prev_h)) + self.bh)
        
        # New hidden state
        h = (1 - z) * prev_h + z * h_hat
        
        cache = (x, prev_h, z, r, h_hat)
        return h, cache

# Example usage
def example_usage():
    # Initialize parameters
    input_size = 10
    hidden_size = 20
    batch_size = 1
    
    # Create input data
    x = np.random.randn(input_size, batch_size)
    prev_h = np.zeros((hidden_size, batch_size))
    prev_c = np.zeros((hidden_size, batch_size))
    
    # Initialize cells
    lstm = LSTM(input_size, hidden_size)
    gru = GRU(input_size, hidden_size)
    
    # Forward pass through LSTM
    lstm_h, lstm_c, lstm_cache = lstm.forward(x, prev_h, prev_c)
    print("LSTM hidden state shape:", lstm_h.shape)
    print("LSTM cell state shape:", lstm_c.shape)
    
    # Forward pass through GRU
    gru_h, gru_cache = gru.forward(x, prev_h)
    print("GRU hidden state shape:", gru_h.shape)

if __name__ == "__main__":
    example_usage()
