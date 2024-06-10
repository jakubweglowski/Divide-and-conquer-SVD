import numpy as np

# Tworzenie macierzy dwudiagonalnej:

np.random.seed(0)
q = np.random.rand(5) # główna diagonala
r = np.random.rand(4) # naddiagonala
B1 = np.diag(q)
B2 = np.diag(r, k=1)
B = B1 + B2
B

##############################################################
# WŁAŚCIWY ALGORYTM:

def DACSVD(B: np.array) -> tuple[np.array]:
    
    if B.shape[1] == 1:
        return(1, B, 1)
    
    # 1: Znajdowanie podziału macierzy B na B1, B2
    m = B.shape[0]//2 # wiersz podziału
    B1 = B[:m, :(m+1)]
    B2 = B[(m+1):, (m+1):]
    qm = B[m, m]
    rm = B[m, m+1]

    # 2: rekurencja
    U1, S1, V1T = DACSVD(B1)
    U2, S2, V2T = DACSVD(B2)
    
    # 3: obliczenie "z" oraz "D"
    l1 = V1T[:-1, -1]
    nu = V1T[-1, -1]
    f2 = V2T[:, 0]
    
    z = np.hstack((nu, qm*l1, rm*f2))
    D = np.diag(np.hstack((0, S1, S2)))

    # 4: rozwiązanie pełnego zadania własnego macierzy D^2 + zz^T