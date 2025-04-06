import numpy as np


def hysime(hsi, n, Rn):
    """Hyperspectral signal subspace estimation."""
    L, N = hsi.shape
    Ln, Nn = n.shape
    d1, d2 = Rn.shape
    if not hsi.size:
        raise ValueError("The data set is empty")
    if Ln != L or Nn != N:
        raise ValueError("Empty noise matrix or its size does not agree with size of y")
    if d1 != d2 or d1 != L:
        print("Bad noise correlation matrix")
        Rn = np.dot(n, n.T) / N

    x = hsi - n
    Ry = (hsi @ hsi.T) / N
    Rx = (x @ x.T) / N
    U, dx, Vt = np.linalg.svd(Rx, full_matrices=False)
    E = Vt.T
    Rn += np.sum(np.diag(Rx)) / L / 1e5 * np.eye(L)
    Py = np.diag(E.T @ Ry @ E)
    Pn = np.diag(E.T @ Rn @ E)
    cost_F = -Py + 2 * Pn
    kf = np.sum(cost_F < 0)
    ind_asc = np.argsort(cost_F)
    Ek = E[:, ind_asc[:kf]]
    return kf, E

