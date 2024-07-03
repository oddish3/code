import numpy as np
from scipy.linalg import sqrtm, svd, eig

def Jhat(PP, BB, CJ, CK, TJ, M, n, nL):
    lb = np.zeros(nL + 1)
    ub = np.zeros(nL + 1)
    
    for ll in range(nL + 1):
        try:
            s = shat(PP[:, CJ[ll]:CJ[ll + 1]], BB[:, CK[ll]:CK[ll + 1]])
        except:
            s = 1e-20
            
        J = TJ[ll]
        lb[ll] = J * np.sqrt(np.log(J)) * max(0 * (np.log(n))**4, 1 / s)
    
    ub[:nL] = lb[1:nL + 1]
    
    L = np.where((lb <= 2 * M * np.sqrt(n)) & (2 * M * np.sqrt(n) <= ub))[0]
    f = 0
    
    if L.size == 0:
        L = np.where(lb <= 2 * M * np.sqrt(n))[0]
        if L.size > 0:
            L = L[-1]
            f = 1
        else:
            L = [0]
            f = 2
    
    LL = max(L[0], 1)
    flag = f
    
    return LL, flag

def shat(P, B):
    Gp = P.T @ P
    Gb = B.T @ B
    S = B.T @ P
    
    if np.min(eig(Gb)[0]) > 0:
        ss = svd(np.linalg.solve(sqrtm(Gb), S) @ np.linalg.inv(sqrtm(Gp)))[1]
        s = np.min(ss)
    else:
        s = 1e-20
        
    return s
