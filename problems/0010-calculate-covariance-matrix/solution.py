def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
    num_features = len(vectors)
    N = len(vectors[0])
    cov_matrix = []
    for i in range(0, num_features):
        mean_i = float(sum(vectors[i]))/N
        row_i = []
        for j in range(0,num_features):
            cov_i_j = 0
        
            mean_j = float(sum(vectors[j]))/N 
            for k in range(0, N):
                cov_i_j += (vectors[i][k]-mean_i) * (vectors[j][k] - mean_j)
            cov_i_j = float(cov_i_j)
            cov_i_j /= (N-1)
            row_i.append(cov_i_j)
        cov_matrix.append(row_i)    

    return cov_matrix