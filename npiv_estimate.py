import numpy as np
from npiv import npiv

def npiv_estimate(Ltil, Px, PP, BB, CJ, CK, y, n):
    Ltil += 1

    Px1 = Px[:, CJ[Ltil]:CJ[Ltil + 1]]
    PP1 = PP[:, CJ[Ltil]:CJ[Ltil + 1]]
    BB1 = BB[:, CK[Ltil]:CK[Ltil + 1]]

    c1, u1, Q1 = npiv(PP1, BB1, y)

    Bu1 = BB1 * u1
    OL1 = Q1 @ (Bu1.T @ Bu1 / n) @ Q1.T

    tden = np.zeros(Px.shape[0])
    for x in range(Px.shape[0]):
        s1 = Px1[x, :] @ OL1 @ Px1[x, :].T
        tden[x] = np.sqrt(s1)

    hhat = Px1 @ c1
    sigh = tden / np.sqrt(n)
    
    return hhat, sigh