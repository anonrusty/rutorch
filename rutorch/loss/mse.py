import numpy as np

class MSELoss:
    def __init__(self):
        self.y_pred = None
        self.y_true = None
        
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        loss = np.mean((self.y_pred - self.y_true)**2)
        return loss
        
    def backward(self):
        n = self.y_true.size
        grad = (2/n) * (self.y_pred - self.y_true)
        return grad
