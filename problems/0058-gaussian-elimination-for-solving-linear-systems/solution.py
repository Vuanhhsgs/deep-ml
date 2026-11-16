import numpy as np

def gaussian_elimination(A, b):
    """
    Solves the system Ax = b using Gaussian Elimination with partial pivoting.
    
    :param A: Coefficient matrix
    :param b: Right-hand side vector
    :return: Solution vector x
    """
    tol = 1e-09
    m = A.shape[0]
    n = A.shape[1]
    A = A.astype(float)
    b = b.astype(float)
    for i in range(0, min(m,n)):
        max = A[i][i]
        max_index = i
        for j in range(i, m):
            if(A[j][i] > max):
                max = A[j][i]
                max_index = j
        temp_A = A[i].copy()
        A[i] = A[max_index]
        A[max_index] = temp_A 

        temp_b = b[i].copy()
        b[i] = b[max_index]
        b[max_index] = temp_b

        for j in range(i+1,m):
            if abs(A[j][i]) <= tol:
                continue 
            else:
                elim_coefficient = A[j][i]/A[i][i]
                A[j] -= A[i] * elim_coefficient
                b[j] -= b[i] * elim_coefficient

    r = min(m,n)
    x = np.zeros_like(b)
    x[r-1] = b[r-1]/A[r-1][r-1]
    for i in range(r-2, -1, -1):
        sum=b[i]
        for j in range(r-1,i,-1):
            sum -= x[j] * A[i][j]
        x[i] = sum/A[i][i]
    return x

    

        
