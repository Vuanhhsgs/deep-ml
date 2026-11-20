import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
    a = np.array(a)
    count = a.shape[0] * a.shape[1]
    if(count != new_shape[0] * new_shape[1]):
        return []
    new_matrix = a.reshape((new_shape[0], new_shape[1]))


    return new_matrix