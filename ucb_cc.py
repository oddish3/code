import numpy as np
from npiv import npiv

def ucb_cc(L, Px, PP, BB, CJ, CK, y, n, nb, type, alpha):
    omega = np.random.randn(n, nb)

    # Step 1: compute critical value
    z = np.zeros(nb)
    i = L + 1

    # Precompute that which can be pre-computed
    Px1 = Px[:, CJ[i]:CJ[i + 1]]
    PP1 = PP[:, CJ[i]:CJ[i + 1]]
    BB1 = BB[:, CK[i]:CK[i + 1]]

    _, u1, Q1 = npiv(PP1, BB1, y)

    Bu1 = BB1 * u1

    OL1 = Q1 @ (Bu1.T @ Bu1 / n) @ Q1.T

    # Variance term
    tden = np.zeros(Px.shape[0])
    for x in range(Px.shape[0]):
        s1 = Px1[x, :] @ OL1 @ Px1[x, :].T
        tden[x] = np.sqrt(s1)

    # Bootstrap
    for b in range(nb):
        Buw1 = Bu1.T @ omega[:, b] / np.sqrt(n)
        
        # Compute bootstrapped sup-t-stat at (J, J2)
        tnum = Px1 @ Q1 @ Buw1
        if type == 0:
            z[b] = np.max(np.abs(tnum / tden))
        elif type == -1:
            z[b] = np.max(tnum / tden)
        elif type == 1:
            z[b] = np.min(tnum / tden)

    # Critical value
    if type == 0 or type == -1:
        cv = np.quantile(z, 1 - alpha)
    elif type == 1:
        cv = -np.quantile(z, alpha)
    
    return cv