import numpy as np

def calculate_correlation_matrix(X, Y=None):
    sample_size = X.shape[0]
    X_num_features = X.shape[1]
    if(Y is None):
        Y = X.copy()
    X = X.astype(float)
    Y = Y.astype(float)
    Y_num_features = Y.shape[1]
    

    X_mean = X.sum(axis = 0)/sample_size
    Y_mean = Y.sum(axis = 0)/sample_size

    X -= X_mean
    Y -= Y_mean
    #cov_X_Y_(i,j) = cov(X_i, Y_j) = (1/N-1)* (X_i-X_mean_i)^T . (Y_j - Y_mean)
    #=sum(k=1 to N) X'_{k,i} * Y'_{k,j} = sum(k=1 to N) ( (X')^T )_{i,k} * Y'_{k,j} = ((X')^T.Y')_{i,j}
    #therefore cov_X_Y = (X')^T . Y'

    #corr_X_Y_(i,j) = corr(X_i, Y_j) = cov(X_i, Y_j)/ ( sqrt(sum of X'_i^2) ) * ( sqrt(sum of Y'_j^2) ) 
    # = ((X')^T.Y')_{i,j} / (X''_i) * (Y''_j) = ((X')^T.Y')_{i,j} / ((X'')^T . Y'' )_{i,j}

    cov_X_Y = X.T @ Y

    X_sq = X*X
    Y_sq = Y*Y 
    var_X = X_sq.sum(axis = 0)
    var_Y = Y_sq.sum(axis = 0)

    sd_X = np.sqrt(var_X)
    sd_Y = np.sqrt(var_Y)

   

    norm_constant_matrix = sd_X.reshape(-1,1) @ sd_Y.reshape(1,-1)

    corr_matrix = cov_X_Y/norm_constant_matrix


    return corr_matrix

    




    