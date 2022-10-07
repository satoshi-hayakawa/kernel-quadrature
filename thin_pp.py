import numpy as np
from numpy.lib.arraysetops import unique
import numpy.random as npr
from argparse import ArgumentParser

import pathlib
import os
import os.path
import pickle as pkl

# goodpoints imports
from goodpoints import kt, compress


from functools import partial


def sob(y, s):
    x = np.maximum(0, y) + (1 + np.minimum(0, y)) * (y < 0)
    tmp = np.zeros(y.shape)
    if s == 1:
        tmp = 1 + 2 * (np.pi ** 2) * ((x ** 2) - x + 1 / 6)
    if s == 2:
        tmp = 1 - (np.pi ** 4) * 2 / 3 * \
            ((x ** 4) - 2 * (x ** 3) + (x ** 2) - 1 / 30)
    if s == 3:
        tmp = 1 + (np.pi ** 6) * 4 / 45 * ((x**6) - 3 * (x**5) +
                                           5 / 2 * (x**4) - (x ** 2) / 2 + 1 / 42)
    return np.prod(tmp, axis=-1)


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
    if params_k["name"] in ["sob"]:
        k_vals = sob(x - y, params_k["var"])
        return(k_vals)

    raise ValueError("Unrecognized kernel name {}".format(params_k["name"]))


def compute_params_k(var_k, d,  use_krt_split=1, name="gauss"):
    '''
        return parameters, and functions for split and swap kernel;
        parameters returned should be understood by kernel_eval, p_kernel,
        ppn_kernel, and pp_kernel functions

        var_k: float scale for the kernel
        d: dimensionality of the problem
        use_krt_split: Whether to use krt for split
        setting: which kernel to use; kernel_eval needs to be defined for that setting
    '''
    params_k_swap = {"name": name, "var": var_k, "d": int(d)}
    if name == "gauss":
        if use_krt_split != 0:
            params_k_split = {"name": "gauss_rt", "var": var_k/2., "d": int(d)}
        else:
            params_k_split = {"name": "gauss", "var": var_k, "d": int(d)}
    else:
        params_k_split = params_k_swap

    split_kernel = partial(kernel_eval, params_k=params_k_split)
    swap_kernel = partial(kernel_eval, params_k=params_k_swap)
    return(params_k_split, params_k_swap, split_kernel, swap_kernel)


def main(X, lsize, lam, name='gauss'):
    _, d = X.shape
    args_size = lsize
    args_g = args_size if args_size <= 4 else 4
    args_krt = True
    args_symm1 = True
    assert(args_g <= args_size)

    ####### seeds #######
    seed_sequence = np.random.SeedSequence()
    seed_sequence_children = seed_sequence.spawn(2)

    thin_seeds_set = seed_sequence_children[0].generate_state(1000)
    compress_seeds_set = seed_sequence_children[1].generate_state(1000)

    # define the kernels
    params_k_split, params_k_swap, split_kernel, swap_kernel = compute_params_k(d=d, var_k=lam,
                                                                                use_krt_split=args_krt, name=name)

    # Specify base failure probability for kernel thinning
    delta = 0.5
    # Each Compress Halve call applied to an input of length l uses KT( l^2 * halve_prob )
    halve_prob = delta / (4*(4**args_size)*(2**args_g) *
                          (args_g + (2**args_g) * (args_size - args_g)))
    ###halve_prob = 0 if size == g else delta * .5 / (4 * (4**size) * (4 ** g) * (size - g) ) ###
    # Each Compress++ Thin call uses KT( thin_prob )
    thin_prob = delta * args_g / (args_g + ((2**args_g)*(args_size - args_g)))
    ###thin_prob = .5

    thin_seed = thin_seeds_set[0]
    compress_seed = compress_seeds_set[0]

    halve_rng = npr.default_rng(compress_seed)

    if args_symm1:
        halve = compress.symmetrize(lambda x: kt.thin(X=x, m=1, split_kernel=split_kernel,
                                    swap_kernel=swap_kernel, seed=halve_rng, unique=True, delta=halve_prob*(len(x)**2)))
    else:
        def halve(x): return kt.thin(X=x, m=1, split_kernel=split_kernel,
                                     swap_kernel=swap_kernel, seed=halve_rng, delta=halve_prob*(len(x)**2))

    thin_rng = npr.default_rng(thin_seed)

    thin = partial(kt.thin, m=args_g, split_kernel=split_kernel, swap_kernel=swap_kernel,
                   seed=thin_rng, delta=thin_prob)

    coreset = compress.compresspp(X, halve, thin, args_g)
    return coreset
