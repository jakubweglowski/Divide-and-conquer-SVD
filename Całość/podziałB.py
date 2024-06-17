import numpy as np

def divideB(B: np.array) -> tuple[np.array]:  
    k = B.shape[1]//2 # kolumna podziału
    
    alphak = B[k, k]
    betak = B[k+1, k]
    B1 = B[:(k+1), :k]
    B2 = B[(k+1):, (k+1):]
    
    return B1, B2, alphak, betak, k