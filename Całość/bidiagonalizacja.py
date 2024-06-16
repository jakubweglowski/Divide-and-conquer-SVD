import numpy as np
from numpy.linalg import norm

def bidiagonalize(A: np.array) -> tuple[np.array]:
    
    B = np.copy(A)
    M, N = B.shape
    
    transposed = False
    if M < N:
        transposed = True
        B = B.T
        
    U0T = np.eye(M)
    V0 = np.eye(N)
    
    # print(f"Przed wejściem do pętli: B=\n{B}\n")
    for i in range(N):
        
        # print(f"KROK {i}\n")
        
        # działanie w kolumnie
        u = np.copy(B[i:, i])
        k = len(u)
        u[0] += norm(u)
        v = (np.zeros_like(u) if np.isclose(norm(u), 0) else u/norm(u))
        Hi = np.eye(k) - 2 * v.reshape((k, 1)) @ v.reshape((1, k))
        Hleft = np.block([[np.eye(i), np.zeros((i, k))],
                        [np.zeros((k, i)), Hi]])
        
        # print(f"Hleft=\n{Hleft}\n")
        
        U0T = (Hleft @ U0T)

        # print(f"Hleft * B=\n{Hleft @ B}\n")
        # print(f"U0T * A=\n{U0T @ A}\n")
        
        # B[i:, i] = Hi @ B[i:, i]
        B = (Hleft @ B)
        
        if i == N-1:
            break
        
        # działanie w wierszu
        u = np.copy(B[i, (i+1):])
        k = len(u)
        u[0] += norm(u)
        v = (np.zeros_like(u) if np.isclose(norm(u), 0) else u/norm(u))
        Hi = np.eye(k) - 2 * v.reshape((k, 1)) @ v.reshape((1, k))
        Hright = np.block([[np.eye(i+1), np.zeros((i+1, k))],
                    [np.zeros((k, i+1)), Hi]])
        
        # print(f"Hright=\n{Hright}\n")
        
        V0 = (V0 @ Hright)
        
        # print(f"B * Hright=\n{B @ Hright}\n")
        B = (B @ Hright)
        # print(f"A * V0=\n{A @ V0}\n")
        
        # B[i, (i+1):] = B[i, (i+1):] @ Hi
        
    assert np.allclose(U0T.T @ B @ V0.T, A)
        
    return U0T.T, B, V0.T, transposed