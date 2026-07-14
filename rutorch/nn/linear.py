import numpy as np

class Linear:
    def __init__(self, input_size, output_size):
        self.W = np.random.randn(input_size,output_size) * 0.01
        self.b = np.random.randn(1,output_size)
    
    def forward(self, x):
        self.x = x
        y = x @ self.W + self.b
        return y
	
    def backward(self, grad_output, learning_rate=0.01):
        grad_W = self.x.T @ grad_output
        grad_b = np.sum(grad_output, axis=0, keepdims=True)

        self.W -= learning_rate * grad_W
        self.b -= learning_rate * grad_b

        return grad_output @ self.W.T
