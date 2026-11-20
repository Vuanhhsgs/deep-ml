import math
import numpy as np
def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
    X = np.array(features)
    X = X.T
    feature_num = X.shape[1]
    y = np.array(labels)
    W = np.array(weights).reshape((1, -1))
    Z = W@X + bias

    T = 1/(1+np.exp(-1*Z))
    

    probabilities = np.round(T, decimals=4)

    mse = probabilities-y
    mse = mse * mse 
    mse = np.sum(mse, axis = 1)
    mse /= feature_num





    return probabilities, mse