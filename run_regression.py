import numpy as np
import os
from bspline import bspline
from jhat import Jhat
from jlep import Jlep
from npiv_estimate import npiv_estimate
from ucb_cv import ucb_cv
from ucb_cvge import ucb_cvge
from ucb_cc import ucb_cc

# Main file to run regression simulations

array_value = 1 # Example value, should be set appropriately
if array_value <= 4:
    trimming = True
    n_index = array_value
else:
    trimming = False
    n_index = array_value - 4

print(trimming)
print(n_index)

# Inputs
nn = [1250, 2500, 5000, 10000]  # sample sizes
nm = 1000                       # number of replications
nb = 1000                       # number of bootstrap draws per replication
nx = 1000                       # number of points for grid of x values
nL = 9                          # maximum resolution level for J
r = 4                           # B-spline order
M = 5                           # ex ante upper bound on \sup_x h_0(x)
alpha = [0.10, 0.05, 0.01]      # level of significance
nj = 4

# Pre-compute
TJ = 2 ** (np.arange(nL + 1)) + r - 1
CJ = np.concatenate(([0], np.cumsum(TJ)))
Lhat = np.zeros(nm)
Llep = np.zeros(nm)
Ltil = np.zeros(nm)
flag = np.zeros(nm)
thet = np.zeros(nm)
zast = np.zeros((nm, len(alpha)))
zdet = np.zeros((nm, len(alpha), nj))
cvge = np.zeros((nm, len(alpha), nj + 1))
loss = np.zeros((nm, nj + 1))
rati = np.zeros((nm, nj))

Xx = np.linspace(0, 1, nx + 1).reshape(-1, 1)
if trimming:
    Xx_sub = Xx[(Xx > 0.01) & (Xx <= 0.99)]
else:
    Xx_sub = Xx

Px = np.zeros((len(Xx_sub), CJ[-1]))
for ll in range(nL + 1):
    Px[:, CJ[ll]:CJ[ll + 1]] = bspline(Xx_sub, ll, r)[0]

h0 = np.sin(15 * np.pi * Xx) * np.cos(Xx)

# Simulations
np.random.seed(1234567)

n = nn[n_index - 1]

for j in range(nm):
    
    if j % 25 == 0:
        print(f'j = {j}')
    if j == nm - 1:
        print('\n')
    
    # Simulate data
    x = np.random.rand(n, 1)
    u = np.random.randn(n, 1)
    y = np.sin(15 * np.pi * x) * np.cos(x) + u
    
    # Pre-compute basis functions and store in arrays PP and BB
    PP = np.zeros((n, CJ[-1]))
    for ll in range(nL + 1):
        PP[:, CJ[ll]:CJ[ll + 1]] = bspline(x, ll, r)
    
    # Compute \hat{J}_{\max} resolution level
    Lhat[j], flag[j] = Jhat(PP, PP, CJ, CJ, TJ, M, n, nL)
    
    # Compute Lepski method resolution level
    Llep[j], thet[j] = Jlep(Lhat[j], Px, PP, PP, CJ, CJ, TJ, y, n, nb)
    
    # Compute \tilde{J} resolution level
    Ltil[j] = max(min(Llep[j], Lhat[j] - 1), 0)
    
    # Compute estimator and pre-asymptotic standard error
    hhat, sigh = npiv_estimate(Ltil[j], Px, PP, PP, CJ, CJ, y, n)
    
    # Compute critical value for UCB
    zast[j, :] = ucb_cv(Ltil[j], Lhat[j], Px, PP, PP, CJ, CJ, y, n, nb, 0, alpha)
    
    # Compute sup-norm loss and excess width
    loss[j, 0] = np.max(np.abs(h0[np.isin(Xx, Xx_sub)] - hhat))
    
    # Compute coverage
    #cvge[j, :, 0] = ucb_cvge(h0[np.isin(Xx, Xx_sub)], hhat, sigh, zast[j, :], thet[j], np.log(np.log(TJ[Llep[j] + 1])))
    
    for k in range(nj):
        
        # Compute undersmoothed estimator and pre-asymptotic standard error
        hha1, sig1 = npiv_estimate(k + 2, Px, PP, PP, CJ, CJ, y, n)
        
        # Compute deterministic J critical value for undersmoothed UCB
        zdet[j, :, k] = ucb_cc(k + 2, Px, PP, PP, CJ, CJ, y, n, nb, 0, alpha)
        
        # Compute sup-norm loss and excess width
        #loss[j, 1 + k] = np.max(np.abs(h0[np.isin(Xx, Xx_sub)] - hha1))
        #rati[j, k] = np.max(sig1) / np.max(sigh)
        
        # Compute coverage
        #cvge[j, :, 1 + k] = ucb_cvge(h0[np.isin(Xx, Xx_sub)], hha1, sig1, zdet[j, :, k], 0, 0)
        
# Save results
np.savez(f'./results/regression_{array_value}.npz', loss=loss, cvge=cvge, rati=rati, zdet=zdet, zast=zast, Llep=Llep, Ltil=Ltil, Lhat=Lhat, thet=thet, TJ=TJ)
