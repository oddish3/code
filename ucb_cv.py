import numpy as np
from scipy.stats import norm
from npiv import npiv

def ucb_cv(Ltil, Lhat, Px, PP, BB, CJ, CK, y, n, nb, type, alpha):
    Lmax = Lhat + 1
    omega = np.random.randn(n, nb)
    ZZ = np.zeros((max(Ltil, Lmax - 2), nb))

    for i in range(max(Ltil, Lmax - 2)):
        
        Px1 = Px[:, CJ[i]:CJ[i + 1]]
        PP1 = PP[:, CJ[i]:CJ[i + 1]]
        BB1 = BB[:, CK[i]:CK[i + 1]]
        
        _, u1, Q1 = npiv(PP1, BB1, y)
        
        Bu1 = BB1 * u1
        
        OL1 = Q1 @ (Bu1.T @ Bu1 / n) @ Q1.T
        
        tden = np.zeros(Px.shape[0])
        for x in range(Px.shape[0]):
            s1 = Px1[x, :] @ OL1 @ Px1[x, :].T
            tden[x] = np.sqrt(s1)
        
        for b in range(nb):
            Buw1 = Bu1.T @ omega[:, b] / np.sqrt(n)
            tnum = Px1 @ Q1 @ Buw1
            
            if type == 0:
                ZZ[i, b] = np.max(np.abs(tnum / tden))
            elif type == -1:
                ZZ[i, b] = np.max(tnum / tden)
            elif type == 1:
                ZZ[i, b] = np.min(tnum / tden)

    if type == 0 or type == -1:
        z = np.max(ZZ, axis=0)
        cv = np.quantile(z, 1 - alpha)
    elif type == 1:
        z = np.min(ZZ, axis=0)
        cv = -np.quantile(z, alpha)
    
    return cv