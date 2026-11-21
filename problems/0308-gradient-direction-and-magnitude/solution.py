import numpy as np
import math
def gradient_direction_magnitude(gradient: list) -> dict:
    if(gradient == [0.0, 0.0]):
        return {'magnitude':0.0, 'direction':[0.0, 0.0]}
    
    gradient = np.array(gradient).reshape(1,-1)

    magnitude = gradient*gradient 
    magnitude = np.sum(magnitude, axis = 1)
    magnitude = np.sqrt(magnitude)

    direction = gradient/magnitude
    descent_direction = -1 * direction 

    ans = {}
    ans['magnitude'] = magnitude.item()

    ans['direction'] = direction.astype(float).tolist()[0]

    ans['descent_direction'] = descent_direction.astype(float).tolist()[0]

    return ans



 

    
    