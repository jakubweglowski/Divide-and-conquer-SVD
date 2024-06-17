import numpy as np

def divideB(B: np.array) -> tuple[np.array]:
    m = (B.shape[0]//2 - 1 if B.shape[0] == B.shape[1] else B.shape[0]//2) # wiersz podziału
    
    # print(f"{B.shape=}\n{B=}")
    qm = B[m, m]
    rm = B[m, m+1]
    B1 = B[:m, :(m+1)]
    B2 = B[(m+1):, (m+1):]
    
    return B1, B2, qm, rm, m