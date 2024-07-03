import numpy as np

def ucb_cvge(h0, hhat, sigh, zast, theta, A):
    loss = np.max(np.abs(hhat - h0))
    tmax = np.max(np.abs((hhat - h0) / sigh))

    check = np.zeros((len(zast), len(A)))

    for i in range(len(zast)):
        for j in range(len(A)):
            check[i, j] = (tmax <= zast[i] + A[j] * theta)

    return check, loss
