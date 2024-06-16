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

import bidiagonalizacja
reload(bidiagonalizacja)

from przypadek1 import *
from przypadek2 import *
from przypadek3 import *
from podziałB import *
from bidiagonalizacja import *

##############################################################
def enforce_zeros(A: np.array) -> np.array:
    A[np.where(np.abs(A) < 1e-15)] = 0
    return A

##############################################################
# WŁAŚCIWY ALGORYTM:

def DACSVD(A: np.array) -> tuple[np.array]:
    
    U0, B, V0T, transposed = bidiagonalize(A)
    Mb, Nb = B.shape
    
    if transposed:
        print(f"Macierz A ma więcej kolumn niż wierszy.\nObliczamy SVD macierzy transponowanej.")
    
    B = enforce_zeros(B)
    
    Q, S, WT = DACSVD_bidiagonal(B[:Nb, :Nb])
    
    Mq, Nq = Q.shape
    Qfull = np.block([[Q, np.zeros((Mq, Mb-Nb))],
                      [np.zeros((Mb-Nb, Nq)), np.eye(Mb-Nb)]])
    Sfull = np.vstack((np.diag(S), np.zeros((Mb-Nb, S.shape[0]))))
    
    if not transposed:
        return U0 @ Qfull, Sfull, WT @ V0T
    else:
        return (V0T @ WT).T, Sfull, (Qfull @ U0).T
    
    
def DACSVD_bidiagonal(B: np.array) -> tuple[np.array]:

    if B.shape[0] <= 3:
        return np.linalg.svd(B)
    
    # 1: Znajdowanie podziału macierzy B na B1, B2
    B1, B2, qm, rm, m = divideB(B)

    # 2: rekurencja
    U1, S1, V1T = DACSVD_bidiagonal(B1)
    U2, S2, V2T = DACSVD_bidiagonal(B2)
    
    # 3: obliczenie "Pm", "z", "D"
    l1 = V1T[:-1, -1]
    nu = V1T[-1, -1]
    f2 = V2T[:, 0]
    
    N = 1 + len(S1) + len(S2)
    Pm = np.eye(N)
    Pm[[0, m], :] = Pm[[m, 0], :]
    
    C = np.block([[np.diag(S1), np.zeros((S1.shape[0], 1)), np.zeros((S1.shape[0], S2.shape[0]))],
                  [qm*l1, qm*nu, rm*f2],
                  [np.zeros((S2.shape[0], S1.shape[0])), np.zeros((S2.shape[0], 1)), np.diag(S2)]])
    
    M = Pm @ C @ Pm.T
    
    z = M[0, :]
    D = np.diag(np.hstack((0, np.diag(M)[1:])))
    
    D2 = np.copy(D)**2
    

    # 4: rozwiązanie pełnego zadania własnego macierzy M^T * M = D^2 + z * z^T  
    H = case_two(D2, z)
    D1, Dnew, znew, P = case_one(D2, H @ z)
    
    nzeros = D1.shape[0]
    Nnew = Dnew.shape[0]
    
    ### sortujemy diagonalę Dnew i wektor znew
    Pd = np.eye(N)[np.argsort(np.diag(Dnew))]
    Dnew = Pd @ Dnew @ Pd.T
    znew = Pd @ znew
    S, Q = case_three(Dnew, znew)
    
    # print("Krok 3...", end=" ")
    assert np.allclose(Dnew + znew.reshape((Nnew, 1)) @ znew.reshape((1, Nnew)), Q @ S @ np.linalg.inv(Q))
    # print("Ok!")

    U = np.block([
        [np.eye(nzeros), np.zeros((nzeros, Nnew))],
        [np.zeros((Nnew, nzeros)), Pd.T @ Q]
    ])

    # elementy na przekątnej Sigma to wartości własne M^T * M
    Sigma = np.block([
        [D1, np.zeros((nzeros, Nnew))],
        [np.zeros((Nnew, nzeros)), S]
    ])
    
    # kolumny Ubar to wektory własne M^T * M
    Ubar = H.T @ P.T @ U
    
    # w tym momencie mamy poprawny rozkład M^T * M = Ubar * Sigma * Ubar^(-1):
    # print("Sprawdzamy rozkład M^T * M = Ubar * Sigma * Ubar^(-1)...", end=" ")
    assert np.allclose(M.T @ M, Ubar @ Sigma @ np.linalg.inv(Ubar))
    # print("Ok!")
    # kolumny Ubar = wektory własne M^T * M
    # diagonala Sigma = wartości własne M^T * M
      
    Lambda = np.sqrt(Sigma)
    Y = Ubar
    
    X = M @ Y @ (np.diag(1/np.diag(Lambda)))

    # print(f"{U1=}\n{U2=}\n")
    # print(f"{V1T=}\n{V2T=}\n")
    U = np.block([[U1, np.zeros((U1.shape[0], 1)), np.zeros((U1.shape[0], U2.shape[1]))],
                  [np.zeros((1, U1.shape[1])), 1, np.zeros((1, U2.shape[1]))],
                  [np.zeros((U2.shape[0], U1.shape[1])), np.zeros((U2.shape[0], 1)), U2]]) @ Pm.T @ X
    
    VT = Y.T @ Pm @ np.block([[V1T, np.zeros((V1T.shape[0], V2T.shape[1]))],
                              [np.zeros((V2T.shape[0], V1T.shape[1])), V2T]])
        
    return U, np.diag(Lambda), VT


##############################################
if __name__ == "__main__":
    M, N = 6, 6
    
    print("Generujemy macierz A...")
    np.random.seed(0)
    A = np.random.rand(M, N)
    
    print("Robimy DACSVD...")
    U, S, VT = DACSVD(A)
    
    print("Upewniamy się, czy A = U*S*VT...", end=" ")
    assert np.allclose(A, U @ S @ VT)
    print("Tak!\n")
    
    print("Teraz robimy SVD z np.linalg i porównujemy:")
    trueSVD = np.linalg.svd(A)
    
    print(f"U=\n{U}\nU_linalg=\n{trueSVD[0]}\n")
    print(f"S=\n{np.diag(S)}\nS_linalg=\n{trueSVD[1]}\n")
    print(f"VT=\n{VT}\nVT_linalg=\n{trueSVD[2]}\n")