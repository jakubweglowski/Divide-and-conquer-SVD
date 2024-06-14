import numpy as np

def case_one(D: np.array, tildez: np.array) -> tuple[np.array]:
    
    # Macierz "D" jest tą samą macierzą "D", co na początku
    # Wektor "tildez" to wektor "Hz", gdzie macierz "H" tworzymy w Przypadku 2.
    
    assert D.shape[0] == D.shape[1]
    assert D.shape[0] == tildez.shape[0]
    assert np.array_equal(D, np.diag(np.diag(D)))

    N = D.shape[0]
    
    A = D + tildez.reshape((N, 1))@tildez.reshape((1, N))

    nzeros = sum(np.isclose(tildez, 0, atol=1e-15)) # liczba zer w "tildez"
    perm = np.isclose(tildez, 0, atol=1e-15) # permutacja przestawiająca zera w "tildez" na początek
    
    # macierz permutacji, o której mowa w notatkach z wykładu:
    P = np.eye(N)
    P = np.vstack((P[perm], P[~perm]))
    
    PDPt = P @ D @ P.T
    PAPt = P @ A @ P.T
    Pz = P @ tildez
    
    Anew = PAPt[nzeros:, nzeros:]
    Dnew = PDPt[nzeros:, nzeros:]
    znew = Pz[nzeros:]
    
    assert np.all(np.isclose(Anew, Dnew + znew.reshape((N-nzeros, 1))@znew.reshape((1, N-nzeros))))
    
    return (Dnew, znew, P, nzeros)

