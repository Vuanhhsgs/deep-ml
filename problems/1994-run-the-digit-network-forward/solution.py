import numpy as np


def digit_network_forward(image, weights, biases):
    layers_X = []
    layers_X.append(np.array(image))
    layers_X[0] = layers_X[0].reshape(-1, 1)
    num_of_layers = len(weights)
  
    for i in range(0, num_of_layers):
        W = np.array(weights[i])
        b = np.array(biases[i]).reshape(-1,1)

        X = W @ layers_X[i] + b 
        X = 1/(1+np.exp(-1*X))
        layers_X.append(X)
    y = np.round(layers_X[-1], 4).reshape(1, -1)
    y = y.tolist()[0]
    max_index = y.index(max(y))

    return (y, max_index)
