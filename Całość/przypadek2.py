import numpy as np

def count_elements(arr: np.array) -> dict:
    
    retdict = {}
    for i, a in enumerate(arr):
        if retdict.get(a) is None:
            retdict[a] = [i]
        else:
            retdict[a].append(i)
    return retdict

def case_two(D: np.array, z: np.array, verbose: bool = False) -> tuple[np.array]:
    
    assert D.shape[0] == D.shape[1]
    assert D.shape[0] == z.shape[0]
    assert np.array_equal(D, np.diag(np.diag(D)))
    
    N = D.shape[0]
    ce = count_elements(np.diag(D))
    
    H = np.eye(N)
    for key, val in ce.items():
            
        if len(val) > 1:
            if verbose: 
                print(f"Blok odpowiadający di={key} na indeksach {val}")
            
            i = val[0]
            k = len(val)
            
            u = z[val]
            u[-1] += np.linalg.norm(u)
            v = u/np.linalg.norm(u)
            Hi = np.eye(k) - 2 * v.reshape((k, 1)) @ v.reshape((1, k))
            
            for row in range(i, i+k):
                for col in range(i, i+k):
                    H[row, col] = Hi[row-i, col-i]
        else:
            if verbose:
                print(f"Pojedyncze di={key} na indeksie {val}.")
    
    return H