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
    A[np.where(np.abs(A) < 1e-14)] = 0
    return A

##############################################################
# WŁAŚCIWY ALGORYTM:

def DACSVD(A: np.array, verbose: bool = False) -> tuple[np.array]:
    
    square = False
    if A.shape[0] == A.shape[1]:
        square = True
        A = np.vstack((A, np.zeros((1, A.shape[1]))))  
        
    U0, B, V0T, transposed = bidiagonalize(A)
    Mb, Nb = B.shape
    
    if verbose and transposed:
        print(f"Macierz A ma więcej kolumn niż wierszy.\nObliczamy SVD macierzy transponowanej.")
    
    B = enforce_zeros(B)
    
    Q, S, WT = DACSVD_bidiagonal(B[:(Nb+1), :], verbose)
    
    Mq, Nq = Q.shape
    Qfull = np.block([[Q, np.zeros((Mq, Mb-Nb-1))],
                      [np.zeros((Mb-Nb-1, Nq)), np.eye(Mb-Nb-1)]])
    Sfull = np.vstack((np.diag(S), np.zeros((Mb-Nb, S.shape[0]))))
    
    if not transposed and not square:
        U, S, VT = U0 @ Qfull, Sfull, WT @ V0T
    elif not transposed and square:
        U, S, VT = (U0 @ Qfull)[:-1, :-1], Sfull[:-1, :], WT @ V0T
    else:
        U, S, VT = (WT @ V0T).T, Sfull.T, (U0 @ Qfull).T
    
    # permutacja żeby s1 >= ... >= sN
    perm = np.argsort(-np.diag(S))
    perm_full = np.hstack((perm, np.arange(len(perm), S.shape[0])))
    S = np.diag(S[:, perm][perm_full, :])
    U = U[:, perm_full]
    VT = VT[perm, :]

    return U, S, VT
    
    
def DACSVD_bidiagonal(B: np.array, verbose: bool = False) -> tuple[np.array]:

    if B.shape[1] <= 2:
        return np.linalg.svd(B)
    
    # 1: Znajdowanie podziału macierzy B na B1, B2
    B1, B2, alphak, betak, _ = divideB(B)
    
    if verbose:
        print(f"Macierz B=\n{B}\nzostała podzielona na:")
        print(f"B1=\n{B1}")
        print(f"B2=\n{B2}")
        print(f"alphak={alphak}")
        print(f"betak={betak}\n")
        
    # 2: rekurencja
    Q1full, D1, W1T = DACSVD_bidiagonal(B1, verbose)
    Q1 = Q1full[:, :-1]
    q1 = Q1full[:, -1]
    lambda1 = q1[-1]
    l1T = Q1[-1, :]
    
    Q2full, D2, W2T = DACSVD_bidiagonal(B2, verbose)
    Q2 = Q2full[:, :-1]
    q2 = Q2full[:, -1]
    phi2 = q2[0]
    f2T = Q2[0, :]
    
    r0 = np.sqrt((alphak*lambda1)**2 + (betak*phi2)**2)
    c0 = alphak*lambda1/r0
    s0 = betak*phi2/r0
    
    if verbose:
        print("Wyznaczamy Q...", end=" ")
    Q = np.block([[c0*q1.reshape((len(q1), 1)), Q1, np.zeros((len(q1), Q2.shape[1]))],
                  [s0*q2.reshape((len(q2), 1)), np.zeros((len(q2), Q1.shape[1])), Q2]])
    q = np.hstack((-s0*q1, c0*q2))
    Qfinal = np.hstack((Q, q.reshape((len(q), 1))))
    if verbose:
        print("Ok!")
        
    if verbose:
        print("Wyznaczamy M...", end=" ")
    M = np.block([[r0, np.zeros((1, len(D1))), np.zeros((1, len(D2)))],
                  [alphak*l1T.reshape((len(l1T), 1)), np.diag(D1), np.zeros((len(D1), len(D2)))],
                  [betak*f2T.reshape((len(f2T), 1)), np.zeros((len(f2T), len(D1))), np.diag(D2)]])
    if verbose:
        print("Ok!")
        
    if verbose:
        print("Wyznaczamy WT...", end=" ")
    WT = np.block([[np.zeros((W1T.shape[1], 1)), W1T.T, np.zeros((W1T.shape[1], W2T.shape[0]))],
                  [1, np.zeros((1, W1T.shape[0])), np.zeros((1, W2T.shape[0]))],
                  [np.zeros((W2T.shape[1], 1)), np.zeros((W2T.shape[1], W1T.shape[0])), W2T.T]]).T
    if verbose:
        print("Ok!")

    # 3: obliczenie "z", "D"
    Mt = np.copy(M.T)
    z = Mt[0, :]
    D = np.diag(np.hstack((0, np.diag(Mt)[1:])))**2
    

    # 4: rozwiązanie pełnego zadania własnego macierzy M * M^T = D^2 + z * z^T
    H = case_two(D, z)
    D1, Dnew, znew, P = case_one(D, H @ z)
    # D1 = enforce_zeros(D1)

    nzeros = D1.shape[0]
    Nnew = Dnew.shape[0]
    
    ### sortujemy diagonalę Dnew i wektor znew
    Pd = np.argsort(np.diag(Dnew))
    Dnew = Dnew[Pd, :][:, Pd]
    znew = znew[Pd]
    Dnew = enforce_zeros(Dnew)
    znew = enforce_zeros(znew)
    S, Q = case_three(Dnew, znew)
    
    # print(f"(Dnew + z * zT)=\n{(Dnew + znew.reshape((Nnew, 1)) @ znew.reshape((1, Nnew)))[0, :]}")
    # print(f"(Q * S * Q^(-1))=\n{(Q @ S @ np.linalg.inv(Q))[0, :]}")
    # print(np.isclose(Dnew + znew.reshape((Nnew, 1)) @ znew.reshape((1, Nnew)), Q @ S @ np.linalg.inv(Q), rtol=1e-3))
    # assert np.allclose(Dnew + znew.reshape((Nnew, 1)) @ znew.reshape((1, Nnew)), Q @ S @ np.linalg.inv(Q), rtol=1e-3)

    U = np.block([
        [np.eye(nzeros), np.zeros((nzeros, Nnew))],
        [np.zeros((Nnew, nzeros)), Q[Pd, :]]
    ])

    # elementy na przekątnej Sigma to wartości własne M^T * M
    Sigma = np.block([
        [D1, np.zeros((nzeros, Nnew))],
        [np.zeros((Nnew, nzeros)), S]
    ])
    
    # kolumny U to wektory własne M * M^T
    U = H.T @ P.T @ U
    
    # w tym momencie mamy poprawny rozkład M * M^T = Ubar * Sigma * Ubar^(-1):
    # if verbose:
    #     print("Sprawdzamy rozkład M * M^T = U * Sigma * U^(-1)...", end=" ")
    # assert np.allclose(M @ M.T, U @ Sigma @ np.linalg.inv(U), atol=1e-1)
    # if verbose:
    #     print("Ok!")
    # kolumny U = wektory własne M * M^T
    # diagonala Sigma = wartości własne M * M^T
    
    # Chcemy svd M = U * S * VT
    S = np.sqrt(Sigma)
    VT = np.diag(1/np.diag(S)) @ U.T @ M
    
    X = Qfinal @ np.block([[U, np.zeros((U.shape[0], 1))],
                           [np.zeros((1, U.shape[1])), 1]])
    
    YT = VT @ WT
    
    return X, np.diag(S), YT


##############################################
if __name__ == "__main__":
    M, N = 1000, 300
    
    print("Generujemy macierz A...")
    A = np.random.rand(M, N)
    
    print("Robimy DACSVD...")
    U, S, VT = DACSVD(A, verbose=False)
    
    # print("Upewniamy się, czy A = U*S*VT...", end=" ")
    # print(np.isclose(A, U @ S @ VT, rtol=1e-3))
    # assert np.allclose(A, U @ S @ VT, rtol=1e-3)
    # print("Tak!\n")
    
    print("Teraz robimy SVD z np.linalg i porównujemy:")
    trueSVD = np.linalg.svd(A)
    
    # print(f"U=\n{U}\nU_linalg=\n{trueSVD[0]}\n")
    # print(f"S=\n{S}\nS_linalg=\n{trueSVD[1]}\n")
    print(f"l2-błąd procentowy na wartościach szczególnych: {100*np.linalg.norm(S - trueSVD[1])/np.linalg.norm(trueSVD[1]) :.4f}%")
    # print(f"VT=\n{VT}\nVT_linalg=\n{trueSVD[2]}\n")