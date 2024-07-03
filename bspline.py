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
    """
    Computes B-spline basis functions and their derivatives.

    Parameters:
    x (numpy array): Points at which to evaluate the B-spline.
    l (int): Level parameter.
    r (int): Degree of the B-spline.
    kts (numpy array, optional): Knot vector. If None, a default knot vector is created.

    Returns:
    numpy array: B-spline basis functions evaluated at x.
    numpy array (optional): Derivatives of B-spline basis functions if requested.
    """
    N = len(x)
    m = 2 ** l - 1
    r = r + 1  # Adjust degree for internal calculations

    # Define the augmented knot set if not provided
    if kts is None:
        if l == 0:
            kts = np.concatenate([np.zeros(r-1), np.ones(r-1)])
        elif l >= 1:
            kts = np.concatenate([np.zeros(r-2), np.linspace(0, 1, 2 ** l + 1), np.ones(r-2)])

    # Initialize for recursion
    BB = np.zeros((N, m + 2*r - 2, r - 1))
    for i in range(N):
        if i == 0 or i == N - 1:  # Print details for the first and last iteration
            print(f"Iteration {i + 1}")
            print(f"x[{i}] = {x[i]}")
            print(f"kts[{r-2}:{r+m-1}] = {kts[r-2:r+m-1]}")
            print(f"kts[{r-1}:{r+m}] = {kts[r-1:r+m]}")

        ix1 = int((x[i] >= kts[r-2:r+m-1]) & (x[i] <= kts[r-1:r+m])) #[0]
    
        if ix1>0:
            ix = ix1
        else:
            ix = None
    
        if i == 0 or i == N - 1:  # Print details for the first and last iteration
            print(f"ix = {ix}")
            print(f"r = {r}")
            print(f"BB[{i}, {ix + r - 2 if ix is not None else 'None'}, 1] = 1")

        if ix is not None:
            BB[i, ix + r - 3, 0] = 1


    # Recursion to compute B-spline basis functions
    for j in range(2, 5):  # j = 2:r-1 in MATLAB
        for i in range(1,9-j):  # i = 1:m+2*r-2-j in MATLAB, adjusting for 0-based indexing
            if i + j <= m + 2*r-1:  # if i+j+1 <= m+2*r in MATLAB, adjusting for 0-based indexing
                if kts[i + j-2] - kts[i-1] != 0:  # kts(i+j-1)-kts(i) in MATLAB
                    a1 = (x - kts[i-1]) / (kts[i + j-2] - kts[i-1])
                else:
                    a1 = np.zeros((N, 1))
            
                if kts[i + j - 1] - kts[i] != 0:  # kts(i+j)-kts(i+1) in MATLAB
                    a2 = (x - kts[i]) / (kts[i + j] - kts[i])
                else:
                    a2 = np.zeros((N, 1))
                a1 = a1.flatten()
                a2 = a2.flatten()
                BB[:, i-1, j-1] = a1 * BB[:, i-1, j - 2] + (1 - a2) * BB[:, i, j - 2]
        
            elif i + j - 1 <= m + 2*r-1:  # if i+j <= m+2*r in MATLAB, adjusting for 0-based indexing
                if kts[i + j - 1] - kts[i-1] != 0:  # kts(i+j)-kts(i) in MATLAB
                    a1 = (x - kts[i-1]) / (kts[i + j -1] - kts[i-1])
                else:
                    a1 = np.zeros((N, 1))
            
                BB[:, i-1, j-1] = a1 * BB[:, i-1, j - 2]
    

    # Compute derivatives if requested
    if nargout() > 1:
        DX = np.zeros_like(XX)
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

        return XX, DX

    return XX


