import numpy as np

def matrix_rank(A: np.ndarray, tol: float = 1e-10) -> int:

    rank = 0
    if(A.shape[0] == 0 and A.shape[1] == 0):
        return 0
    m = A.shape[0]
    n = A.shape[1]
    A = A.astype(float)
    for i in range(0, min(m,n)):
        max = A[i][i]
        max_index = i
        for j in range(i, m):
            if A[j][i] > max:
                max = A[j][i]
                max_index = j
        temp = A[i].copy()
        A[i] = A[max_index]
        A[max_index] = temp 

        if abs(A[i][i]) <= tol:
            continue 
        else:
            rank += 1
            for j in range(i+1, m):
                if abs(A[j][i]) > tol:
                    elim_multiplier = A[j][i]/A[i][i]
                    A[j] -= elim_multiplier * A[i]
                    
    return rank

def is_linearly_independent(vectors: list[list[float]]) -> bool:
    num_of_vectors = len(vectors)
    vectors = np.array(vectors)
    if(num_of_vectors==0):
        return True
    if(matrix_rank(vectors) == num_of_vectors):
        return True
    else:
        return False
    pass