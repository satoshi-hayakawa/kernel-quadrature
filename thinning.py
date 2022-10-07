from goodpoints import compress, kt
from functools import partial
import numpy as np


def kernel_eval(x, y, params_k):
    """Returns matrix of kernel evaluations kernel(xi, yi) for each row index i.
    x and y should have the same number of columns, and x should either have the
    same shape as y or consist of a single row, in which case, x is broadcasted 
    to have the same shape as y.
    """
    if params_k["name"] in ["gauss", "gauss_rt"]:
        k_vals = np.sum((x-y)**2, axis=1)
        scale = -.5/params_k["var"]
        return(np.exp(scale*k_vals))

    raise ValueError("Unrecognized kernel name {}".format(params_k["name"]))


def main(X, m, lam):  # only Gaussian
    var = lam  # Variance
    _, d = X.shape
    # params_p = {"name": "gauss", "var": var,
    #            "d": int(d), "saved_samples": False}

    params_k_swap = {"name": "gauss", "var": var, "d": int(d)}
    params_k_split = {"name": "gauss_rt", "var": var/2., "d": int(d)}

    split_kernel = partial(kernel_eval, params_k=params_k_split)
    swap_kernel = partial(kernel_eval, params_k=params_k_swap)

    coreset = kt.thin(X, m, split_kernel, swap_kernel, delta=0.5, store_K=False,
                      verbose=False)
    return coreset
