import numpy as np
def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
    
    W = initial_weights.copy().reshape(1,-1)
    X = features.copy().T
    b = initial_bias

    y = labels.copy().reshape(1,-1)
    feature_num = y.shape[1]
    mse_values = []
    for i in range(epochs):
        Z = W @ X + b
        M = 1/(1+np.exp(-1*Z))

        L = 1/(feature_num) * (M-y) @ (M-y).T
        L = np.round(L, decimals = 4)
        mse_values.append(L.item())

        dL_dM =  (2/feature_num) *(M-y)
        dL_dZ = dL_dM * ((1-M) * M) #dL_dZ_i = dL_dM_i * f(Z_i) * (1-f(Z_i))
        dL_db = np.sum(dL_dZ, axis = 1)

        b -= dL_db * learning_rate

        dL_dW = dL_dZ @ X.T

        W -= dL_dW * learning_rate

    updated_weights = np.round(W, decimals = 4).astype(float).tolist()[0]
    updated_bias = np.round(b, decimals = 4).item()
    




    return updated_weights, updated_bias, mse_values
    