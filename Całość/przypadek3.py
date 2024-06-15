import numpy as np
from numpy.linalg import solve, norm
from scipy.optimize import bisect

def f(x: float, z: np.array, d: np.array) -> float:
    assert z.shape[0] == d.shape[0]
    return 1 + np.sum(z*z / (d - x))

def case_three(Dnew: np.array, znew: np.array) -> tuple[np.array]:
    
    assert Dnew.shape[0] == Dnew.shape[1]
    assert Dnew.shape[0] == znew.shape[0]
    assert np.array_equal(Dnew, np.diag(np.diag(Dnew)))
    
    Nnew = Dnew.shape[0]
    d = np.diag(Dnew)
    
    # wyznaczamy wartości własne
    eigenvalues = []
    for i in range(Nnew-1):    
        eigenvalues.append(bisect(f, d[i]+(1e-15), d[i+1]-(1e-15), args=(znew, d)))
        
    eigenvalues.append(bisect(f, d[-1]+(1e-15), d[-1] + norm(znew)**2, args=(znew, d)))
    
    # wyznaczamy wektory własne
    eigenvectors = []
    for λ in eigenvalues:
        v = solve(Dnew-λ*np.eye(Nnew), znew)
        eigenvectors.append(v/norm(v))
    
    # składamy macierze
    S = np.diag(eigenvalues)
    Q = np.array(eigenvectors).T
    
    return S, Q