import numpy as np

def npiv(P, B, y):
    Q = np.linalg.pinv(P.T @ B @ np.linalg.pinv(B.T @ B) @ B.T @ P) @ P.T @ B @ np.linalg.pinv(B.T @ B)
    c = Q @ B.T @ y
    uhat = y - P @ c

    QQ = None
    if 'QQ' in locals():
        QQ = Q * len(y)

    return c, uhat, QQ
