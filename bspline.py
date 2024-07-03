import numpy as np
import inspect

def nargout():
    """
    Determines the number of output arguments in the caller's context.
    
    Returns:
    int: Number of output arguments expected.
    """
    frame = inspect.currentframe().f_back
    arg_info = inspect.getargvalues(frame)
    return len(arg_info.locals['__return_values__']) if '__return_values__' in arg_info.locals else 1

def bspline(x, l, r, kts=None):
    N = len(x)
    m = 2 ** l - 1
    r += 1  # Adjust degree for internal calculations

    if kts is None:
        if l == 0:
            kts = np.concatenate([np.zeros(r-1), np.ones(r-1)])
        elif l >= 1:
            kts = np.concatenate([np.zeros(r-2), np.linspace(0, 1, 2 ** l + 1), np.ones(r-2)])

    BB = np.zeros((N, m + 2*r - 2, r - 1))
    for i in range(N):
        ix1 = np.where((x[i] >= kts[r-2:r+m-1]) & (x[i] <= kts[r-1:r+m]))[0]
        ix = ix1[0] if len(ix1) > 0 else None
        if ix is not None:
            BB[i, ix + r - 3, 0] = 1

    for j in range(2, r):
        for i in range(m + 2*r - j - 1):
            if kts[i + j-1] - kts[i] != 0:
                a1 = (x - kts[i]) / (kts[i + j-1] - kts[i])
            else:
                a1 = np.zeros((N, 1))
            if kts[i + j] - kts[i + 1] != 0:
                a2 = (x - kts[i + 1]) / (kts[i + j] - kts[i + 1])
            else:
                a2 = np.zeros((N, 1))
            BB[:, i, j-1] = a1 * BB[:, i, j - 2] + (1 - a2) * BB[:, i + 1, j - 2]

    if nargout() > 1:
        DX = np.zeros((N, m + r - 1))
        for i in range(m + r - 1):
            if kts[i + r - 2] - kts[i] != 0:
                a1 = 1 / (kts[i + r - 2] - kts[i])
            else:
                a1 = np.zeros(N)
            if kts[i + r - 1] - kts[i + 1] != 0:
                a2 = 1 / (kts[i + r - 1] - kts[i + 1])
            else:
                a2 = np.zeros(N)
            if i < m + r - 1:
                DX[:, i] = (r - 2) * (a1 * BB[:, i, r - 2] - a2 * BB[:, i + 1, r - 2])
            else:
                DX[:, i] = (r - 2) * a1 * BB[:, i, r - 2]
        return BB, DX

    return BB


