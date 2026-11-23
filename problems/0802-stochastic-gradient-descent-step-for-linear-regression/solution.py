import numpy as np

def sgd_update(X: np.ndarray, y: np.ndarray, weights: np.ndarray, learning_rate: float, n_iter: int) -> list:
    updated_W = weights
    iteration = 0
    sample_size = X.shape[0]
    while iteration < n_iter:
        j = iteration%sample_size
        dL_dW = 2*( (X[j]@updated_W) - y[j])*X[j]
        updated_W -= learning_rate * dL_dW
        iteration += 1

    return updated_W.astype(float).tolist()

    
