import numpy as np
from numpy.linalg import solve, norm
from scipy.optimize import bisect

def f(x: float, z: np.array, d: np.array, verbose: bool = False) -> float:
    assert z.shape[0] == d.shape[0]
    if verbose and np.any(np.isclose(d-x, 0, atol=1e-14)):
        print(f"Ostrzeżenie z funkcji 'f': możliwy błąd dzielenia przez 0.")
    return (1 + np.sum((z*z) / (d - x)))

def case_three(Dnew: np.array, znew: np.array) -> tuple[np.array]:
    
    assert np.allclose(Dnew, np.diag(np.diag(Dnew)))
    assert Dnew.shape[0] == znew.shape[0]
    
    Nnew = Dnew.shape[0]
    d = np.diag(Dnew)
    
    # wyznaczamy wartości własne
    eigenvalues = []
    for i in range(Nnew-1):
        tol = 0.01*(d[i+1]-d[i])
        succeed = False
        while not succeed and tol > 1e-13:
            try:               
                eigenvalues.append(bisect(f, d[i]+tol, d[i+1]-tol, args=(znew, d)))
                succeed = True
                
            except ValueError as ve:
                # print(f"Kontrolowane przechwycenie błędu: {ve}")
                tol *= 0.8
                
        if not succeed:
            print("Nie można dokładnie wyznaczyć wartości własnej -- zbyt blisko osobliwości.\nPrzyjmujemy najlepsze możliwe przybliżenie.")
            eigenvalues.append(d[i]+1e-10)
            
    # Wyznaczamy ostatnią wartość własną
    tol = 1e-8
    succeed = False
    while not succeed and tol > 1e-13:
        try:            
            eigenvalues.append(bisect(f, d[-1] + tol, d[-1]+norm(znew)**2, args=(znew, d, False)))
            succeed = True
            # print("Sprawdzamy, czy wylądowaliśmy w dobrym przedziale...", end=" ")
            # assert lambda0 > d[-1] and lambda0 < d[-1]+norm(znew)**2
            # print("Ok!")
            
        except ValueError as ve:
            # print(f"Kontrolowane przechwycenie błędu: {ve}")
            tol *= 0.8
            
    if not succeed:
        print("\nNie można dokładnie wyznaczyć wartości własnej -- zbyt blisko osobliwości.\n\t...przyjmujemy najlepsze możliwe przybliżenie.\n")
        eigenvalues.append(d[-1]+1e-10)
        
    # wyznaczamy wektory własne
    eigenvectors = []
    for λ in eigenvalues:
        v = solve(Dnew-λ*np.eye(Nnew), znew)
        eigenvectors.append(-v/norm(v))
    
    # składamy macierze
    S = np.diag(eigenvalues)
    Q = np.array(eigenvectors).T
    
    return S, Q