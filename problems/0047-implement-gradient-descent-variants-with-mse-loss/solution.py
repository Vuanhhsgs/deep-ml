import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    """
    Perform gradient descent optimization.
    
    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')
    
    Returns:
        Optimized weights
    """
    W = weights.reshape(-1,1) 
    y = y.reshape(-1,1)
    
    if method == 'batch':
        m_sample = X.shape[0]
        for i in range(n_epochs):
            dL_dW = 2/m_sample * X.T @ (X@W - y)
            W -= dL_dW * learning_rate
    elif method == 'stochastic':
        m_sample = X.shape[0]
        for i in range(n_epochs):
            for j in range(m_sample):
                dL_dW = 2 * X[j].reshape(1,-1).T * (X[j].reshape(1,-1) @ W - y[j])
                W -= dL_dW * learning_rate
    elif method == 'mini_batch':
        m_sample = X.shape[0]
        num_batch = (m_sample-1)//batch_size + 1 
        for epoch in range(n_epochs):
            for i in range(0, (num_batch-1)*batch_size, batch_size):
                X_batch = []
                y_batch = []
                for j in range(0,batch_size):
                    X_batch.append(X[i+j])
                    y_batch.append(y[i+j])
                X_batch = np.array(X_batch)
                y_batch = np.array(y_batch)
                dL_dW = 2/batch_size * X_batch.T @ (X_batch @ W - y_batch)
                W -= dL_dW * learning_rate
            if(m_sample-1 >= (num_batch-1)*batch_size):
                final_X_batch = []
                final_y_batch = []
                for k in range((num_batch-1)*batch_size, m_sample):
                    final_X_batch.append(X[k])
                    final_y_batch.append(y[k])
                final_X_batch = np.array(final_X_batch)
                final_y_batch = np.array(final_y_batch).reshape(-1,1)
                final_batch_size = m_sample - (num_batch-1)*batch_size 
                dL_dW = 2/final_batch_size * final_X_batch.T @ (final_X_batch@W - final_y_batch)
                W -= dL_dW * learning_rate

    return W.flatten()



    # Your code here
    pass
