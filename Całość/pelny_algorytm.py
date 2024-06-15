import numpy as np

from importlib import reload

import przypadek1
reload(przypadek1)

import przypadek2
reload(przypadek2)

import przypadek3
reload(przypadek3)

import podziałB
reload(podziałB)

from przypadek1 import *
from przypadek2 import *
from przypadek3 import *
from podziałB import *

##############################################################
# WŁAŚCIWY ALGORYTM:

def DACSVD(B: np.array) -> tuple[np.array]:
    
    print(f"\nNowe wywołanie!")
    if B.shape[0] <= 2:
        print(f"Koniec rekurencji! {B=}\n")
        return np.linalg.svd(B)
    
    # 1: Znajdowanie podziału macierzy B na B1, B2
    B1, B2, qm, rm, m = divideB(B)

    print(f"{B=}\nzostała podzielona na\n{B1=}\n{B2=}\n{qm=}\n{rm=}\n")
    # 2: rekurencja
    U1, S1, V1T = DACSVD(B1)
    U2, S2, V2T = DACSVD(B2)
    
    # 3: obliczenie "z" oraz "D"
    l1 = V1T[:-1, -1]
    nu = V1T[-1, -1]
    f2 = V2T[:, 0]
    
    print(f"{B1=}\n{B2=}\n")
    print(f"{S1=}\n{S2=}\n")
    print(f"{U1=}\n{U2=}\n")
    print(f"{V1T=}\n{V2T=}\n")
    print(f"{l1=}\n{nu=}\n{f2=}\n")
    
    z = np.hstack((nu, qm*l1, rm*f2))
    D = np.diag(np.hstack((0, S1, S2)))
    
    print(f"{D=}\n{z=}\n")
    
    M = np.copy(D)
    M[0, :] = z

    # 4: rozwiązanie pełnego zadania własnego macierzy M^T * M = D^2 + z * z^T
    N = D.shape[0]
    A = D + z.reshape((N, 1)) @ z.reshape((1, N))
    
    H = case_two(D, z)
    D1, Dnew, znew, P = case_one(D, H @ z)
    S, Q = case_three(Dnew, znew)
    
    nzeros = D1.shape[0]
    Nnew = Dnew.shape[0]

    U = np.block([
        [np.eye(nzeros), np.zeros((nzeros, Nnew))],
        [np.zeros((Nnew, nzeros)), Q]
    ])

    # elementy na przekątnej Sigma to wartości własne A
    Sigma = np.block([
        [D1, np.zeros((nzeros, Nnew))],
        [np.zeros((Nnew, nzeros)), S]
    ])
    
    # kolumny Ubar to wektory własne A
    Ubar = H.T @ P.T @ U
    
    print(f"Czy A = U*S*U^(-1)?\n{np.all(np.isclose(A, Ubar @ Sigma @ np.linalg.inv(Ubar)))}")
    
    Pm = np.eye(N)
    Pm[[0, m-1], :] = Pm[[m-1, 0], :]
    
    Lambda = np.sqrt(Sigma)
    Y = Ubar
    
    print(f"{Y=}\n{Lambda=}\n")
    X = Y @ (np.diag(1/np.diag(Lambda))) @ M
    
    tempU = np.block([[U1, np.zeros((U1.shape[0], 1)), np.zeros((U1.shape[0], U2.shape[1]))],
                  [np.zeros((1, U1.shape[1])), 1, np.zeros((1, U2.shape[1]))],
                  [np.zeros((U2.shape[0], U1.shape[1])), np.zeros((U2.shape[0], 1)), U2]])
    print(f"{tempU=}\n")
    print(f"{tempU.shape=}\n{Pm.shape=}\n{X.shape=}\n")
    U = tempU @ Pm.T @ X
    
    V = Y @ Pm @ np.block([[V1T, np.zeros((V1T.shape[0], V2T.shape[1]))],
                  [np.zeros((V2T.shape[0], V1T.shape[1])), V2T]])
    
    # if Lambda.shape[0] != Lambda.shape[1]:
    #     Lambda = Lambda[:, :-1]
        
    return U, Lambda, V


##############################################
if __name__ == "__main__":
    # Tworzenie macierzy dwudiagonalnej:

    np.random.seed(0)
    K = 5
    q = np.random.rand(K) # główna diagonala
    r = np.random.rand(K-1) # naddiagonala
    B1 = np.diag(q)
    B2 = np.diag(r, k=1)
    B = B1 + B2
    
    print(f"Zanim wejdziemy do DACSVD:\n{B=}")
    DACSVD(B)