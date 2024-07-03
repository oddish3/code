import numpy as np
from scipy.stats import norm
from npiv import npiv  # Importing the npiv function from npiv.py

def Jlep(Lhat, Px, PP, BB, CJ, CK, TJ, y, n, nb):
    Lhat = int(Lhat)
    Lmax = int(Lhat + 1)
    Jmax = TJ[int(Lhat) + 1]
    
    omega = np.random.randn(n, nb)

    ZZ = np.zeros((Lmax, Lmax, nb))
    HH = np.zeros((Lmax, Lmax))

    for i in range(Lmax):
        for j in range(i + 1, Lmax):
            
            Px1 = Px[:, CJ[i]:CJ[i + 1]]
            PP1 = PP[:, CJ[i]:CJ[i + 1]]
            BB1 = BB[:, CK[i]:CK[i + 1]]
            Px2 = Px[:, CJ[j]:CJ[j + 1]]
            PP2 = PP[:, CJ[j]:CJ[j + 1]]
            BB2 = BB[:, CK[j]:CK[j + 1]]
            
            c1, u1, Q1 = npiv(PP1, BB1, y)
            c2, u2, Q2 = npiv(PP2, BB2, y)
            
            Bu1 = BB1 * u1
            Bu2 = BB2 * u2
            
            OL1 = Q1 @ (Bu1.T @ Bu1 / n) @ Q1.T
            OL2 = Q2 @ (Bu2.T @ Bu2 / n) @ Q2.T
            OL12 = Q1 @ (Bu1.T @ Bu2 / n) @ Q2.T
            
            tden = np.zeros(Px.shape[0])
            for x in range(Px.shape[0]):
                s1 = Px1[x, :] @ OL1 @ Px1[x, :].T
                s2 = Px2[x, :] @ OL2 @ Px2[x, :].T
                s12 = Px1[x, :] @ OL12 @ Px2[x, :].T
                tden[x] = np.sqrt(s1 + s2 - 2 * s12)
            
            tnum = np.sqrt(n) * (Px1 @ c1 - Px2 @ c2)
            HH[i, j] = np.max(np.abs(tnum / tden))
            
            for b in range(nb):
                Buw1 = Bu1.T @ omega[:, b] / np.sqrt(n)
                Buw2 = Bu2.T @ omega[:, b] / np.sqrt(n)
                
                tnum = Px1 @ Q1 @ Buw1 - Px2 @ Q2 @ Buw2
                ZZ[i, j, b] = np.max(np.abs(tnum / tden))

    z = np.zeros(nb)
    for b in range(nb):
        z[b] = np.max(ZZ[:, :, b])

    theta = np.quantile(z, max(0.5, 1 - np.sqrt(np.log(Jmax) / Jmax)))
    
    LL = np.where(np.max(HH, axis=1) <= 1.1 * theta)[0][0] - 1
    
    return LL, theta
