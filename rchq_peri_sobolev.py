import random
import functools
import numpy as np
from scipy.stats.qmc import Halton
import matplotlib.pyplot as plt
import time

import grlp
import emp_mer as emer
import emp_nys as enys

d_global = 1
s_global = 1


def gen_params(n):
    return np.random.rand(n, d_global)


def k(x, y=0, diag=False, s=None):
    if s is None:
        s = s_global
    if np.isscalar(x):
        x = np.array([[x]])
    m, d = x.shape
    if diag:
        return sob(np.zeros((m, d)), s)
    if np.isscalar(y):
        y = np.array([[y]])
    n, _ = y.shape
    X = np.zeros((m, n, d))
    for i in range(n):
        X[:, i, :] += x
    for j in range(m):
        X[j, :, :] -= y
    return sob(X, s)  # edit here to change the kernel


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


def experiments(
    dim=1,
    smooth=1,
    times=50,
    seed=None,
    np_seed=None,
):
    random.seed(seed)
    np.random.seed(np_seed)
    global d_global
    d_global = dim
    global s_global
    s_global = smooth
    enys.k = functools.partial(k, s=smooth)
    if dim == 1:
        emer.k = functools.partial(k, s=smooth)

    text_data = open("results_rchq/d{}s{}t{}.txt".format(
        dim, smooth, times), 'w', encoding='utf-8')
    print("seed = {}, np_seed = {}".format(seed, np_seed), file=text_data)

    fig = plt.figure()
    x_ex = [5, 10, 15, 20, 30, 40, 50, 65, 80]
    #x_ex = [5, 9, 15, 19, 29, 39, 49, 65, 70]
    m_names = ['N. + emp + opt', 'Nyström',
               'Nyström + opt', 'iid Bayes', 'Halton', 'Halton + opt']
    if dim == 1:
        m_names = ['N. + emp + opt', 'Nyström', 'Nyström + opt', 'Mercer',
                   'Mercer + opt', 'Uniform Grid']
    methods = len(m_names)
    results = [[] for i in range(methods)]
    results_up = [[] for i in range(methods)]
    results_low = [[] for i in range(methods)]
    m_marks = ['x', 'o', '^', 'v', '+', '>', '<', 'd', 's']
    for n in x_ex:
        print("{} points. ".format(n), file=text_data)
        for i in range(methods):
            start_time = time.perf_counter()
            res = np.zeros(times)
            fails = 0
            for j in range(times):
                N = n*n
                if 'Mercer' in m_names[i] or 'Nyström' in m_names[i]:
                    N = 10*n
                points, weights, tmp_fails = func(
                    m_names[i], n, rec=N, nys=10*n)
                res[j] = eval(points, weights)
                fails += tmp_fails
            end_time = time.perf_counter()
            elapsed = (end_time - start_time)/times
            res_sq = np.std(res)
            res_mn = np.mean(res)
            res = np.log10(res)
            log_res = np.mean(res)
            log_std = np.std(res)
            results[i].append(np.mean(res))
            results_up[i].append(log_res + log_std)
            results_low[i].append(log_res - log_std)
            print("    {}: {:.2e} (±{:.2e}), {:.2e}s, {} fails".format(
                m_names[i], res_mn, res_sq, elapsed, fails), file=text_data)
    x = np.log10(x_ex)

    for i in range(methods):
        plt.plot(x, results[i], label=m_names[i], marker=m_marks[i])
        plt.fill_between(x, results_low[i], results_up[i], alpha=0.3)
    # plt.xscale("log", nonposx='clip')
    # plt.yscale("log", nonposy='clip')
    plt.legend(loc='lower left', fontsize=12)
    plt.xlabel("$\mathrm{log}_{10} n$", fontsize=20)
    plt.ylabel("$\mathrm{log}_{10} (\mathrm{wce})^2$", fontsize=20)
    plt.tight_layout()
    # plt.show()
    fig.savefig("results_rchq/d{}s{}t{}.pdf".format(dim, smooth, times))
    text_data.close()


def func(name, n, rec=0, nys=0):
    x = []
    w = []
    idx = []
    fails = 0
    if 'N. + emp' in name:
        pts_rec = gen_params(rec)
        pts_nys = gen_params(nys)
        idx, w = enys.recombination(
            pts_rec, pts_nys, n, use_obj=True, rand_SVD=False)
        x = pts_rec[idx]
    elif name == 'iid Bayes':
        x, w = mc_bayes(n)
    elif name == 'Uniform Grid':
        x, w = ug_bayes(n)
    elif 'Halton' in name:
        x = gen_params(1)
        _, d = x.shape
        sampler = Halton(d=d)
        x = sampler.random(n)
        w = np.ones(n)/n
    elif 'Nyström' in name:
        fails = -1
        pts_nys = gen_params(nys)
        svs, U_nys = enys.ker_svd_sparsify(
            pts_nys, n - 1, rand_SVD=False)
        while len(idx) == 0:
            fails += 1
            pts_rec = gen_params(rec)
            idx, w = rchq_nys(pts_rec, pts_nys, n, U_nys, svs, use_obj=True)
        x = pts_rec[idx]
    elif 'Mercer' in name:
        fails = -1
        while len(idx) == 0:
            fails += 1
            pts_rec = gen_params(rec)
            idx, w = rchq_mer(pts_rec, n, use_obj=True)
        x = pts_rec[idx]

    if 'opt' in name:
        if 'Halton' in name:
            w = grlp.QP(k(x, x), k_exp(x), k_exp_exp())
        else:
            w = grlp.QP(k(x, x), k_exp(x), k_exp_exp(), nonnegative=True)
    return x, w, fails


def eval(x, w, pr=False):
    if pr == True:
        print(w)
    if len(x) == 0:
        return 10000000000
    m = len(x)
    tmp = np.transpose(w) @ k_exp(x)
    ret = (k_exp_exp() - tmp) + (np.transpose(w) @ k(x, x) @ w - tmp)
    return ret[0, 0]


def k_exp(x):
    if np.isscalar(x):
        return 1
    else:
        return np.ones((len(x),))


def k_exp_exp():
    return np.ones((1, 1))


def mc(m):
    pt = gen_params(m)
    w = np.ones(m)/m
    return pt, w


def mc_bayes(m, nn=False):
    pt = gen_params(m)
    return pt, grlp.QP(k(pt, pt), k_exp(pt), k_exp_exp(), nonnegative=nn)


def ug_bayes(m):
    pt = np.array([[i / m] for i in range(m)])
    w = np.array([1 / m for _ in range(m)])
    return pt, w
    # return eval(pt, QP(pt, nn))


def rchq_nys(samp, pt, s, U, svs=0, use_obj=False):
    obj = 0
    if use_obj:
        obj = k(samp, diag=True)
        idx_feasible = svs >= 1e-10
        inv_svs = np.zeros(len(svs))
        inv_svs[idx_feasible] = np.sqrt(1/svs[idx_feasible])
        sur_svs = np.reshape(inv_svs, (-1, 1))
        N = len(samp)
        rem = N - s * (N // s)
        for i in range(N//s):
            mat = k(pt, samp[s*i:s*(i+1)])
            mat = U @ mat
            mat = np.multiply(mat, sur_svs)
            obj[s*i:s*(i+1)] -= np.sum(mat**2, axis=0)
        if rem:
            mat = k(pt, samp[N-rem:N])
            mat = U @ mat
            mat = np.multiply(mat, sur_svs)
            obj[N-rem:N] -= np.sum(mat**2, axis=0)
    K = np.ones((s, len(samp)))
    B = np.ones(s,)
    B[:s-1] = U @ k_exp(pt)
    K[:s-1, :] = U @ k(pt, samp)
    sol = grlp.LP(K, B, set_objective=use_obj, obj=obj)
    idx = []
    weights = []
    for a, w_a in sol:
        idx += [a]
        weights += [w_a]

    return idx, weights


def rchq_mer(samp, s, use_obj=False):
    X = emer.eigen_funcs(samp, s - 1)
    obj = 0
    evs = emer.eigen_values(s - 1)
    sur_evs = np.reshape(np.sqrt(evs), (-1, 1))
    if use_obj:
        obj = k(samp, diag=True)
        N = len(samp)
        rem = N - s * (N // s)
        for i in range(N//s):
            mat = X[:, s*i:s*(i+1)]
            mat = np.multiply(mat, sur_evs)
            obj[s*i:s*(i+1)] -= np.sum(mat**2, axis=0)
        if rem:
            mat = X[:, N-rem:N]
            mat = np.multiply(mat, sur_evs)
            obj[N-rem:N] -= np.sum(mat**2, axis=0)

    B = np.zeros(s - 1)
    B[0] = 1

    sol = grlp.LP(X, B, set_objective=use_obj, obj=obj)
    idx = []
    weights = []
    for a, w_a in sol:
        idx += [a]
        weights += [w_a]

    return idx, weights
