import numpy as np
from numpy.linalg import solve, norm
from scipy.optimize import bisect, newton

def f(x: float, z: np.array, d: np.array, verbose: bool = False) -> float:
    assert z.shape[0] == d.shape[0]
    if verbose and np.any(np.isclose(d-x, 0, atol=1e-14)):
        print(f"Ostrzeżenie z funkcji 'f': możliwy błąd dzielenia przez 0...")
    return 1 + np.sum(z*z / (d - x))

def case_three(Dnew: np.array, znew: np.array) -> tuple[np.array]:
    
    assert np.allclose(Dnew, np.diag(np.diag(Dnew)))
    assert Dnew.shape[0] == znew.shape[0]
    
    Nnew = Dnew.shape[0]
    d = np.diag(Dnew)
    
    # wyznaczamy wartości własne
    eigenvalues = []
    for i in range(Nnew-1):
        eigenvalues.append(bisect(f, d[i]+(1e-12), d[i+1]-(1e-12), args=(znew, d)))

    # print(f"f(a)={f(d[-1]+1e-4, znew, d)}\nf(b)={f(d[-1]+norm(znew)**2, znew, d)}\n")
    eigenvalues.append(bisect(f, d[-1]+1e-4, d[-1]+norm(znew)**2, args=(znew, d, True)))
    
    # wyznaczamy wektory własne
    eigenvectors = []
    for λ in eigenvalues:
        v = solve(Dnew-λ*np.eye(Nnew), znew)
        eigenvectors.append(v/norm(v))
    
    # składamy macierze
    S = np.diag(eigenvalues)
    Q = np.array(eigenvectors).T
    
    return S, Q