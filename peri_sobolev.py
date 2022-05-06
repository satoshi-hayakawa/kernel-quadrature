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

    text_data = open("results/d{}s{}t{}.txt".format(
        dim, smooth, times), 'w', encoding='utf-8')
    print("seed = {}, np_seed = {}".format(seed, np_seed), file=text_data)

    fig = plt.figure()
    x_ex = [5, 10, 15, 20, 30, 40, 50, 65, 80]
    m_names = ['N. + emp', 'N. + emp + opt',
               'Monte Carlo', 'iid Bayes', 'Halton', 'Halton + opt']
    if dim == 1:
        m_names = ['N. + emp', 'N. + emp + opt', 'M. + emp',
                   'M. + emp + opt', 'Monte Carlo', 'iid Bayes', 'Uniform Grid']
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
            for j in range(times):
                points, weights = func(m_names[i], n, rec=n*n, nys=10*n)
                res[j] = eval(points, weights)
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
            print("    {}: {:.2e} (±{:.2e}), {:.2e}s".format(
                m_names[i], res_mn, res_sq, elapsed), file=text_data)
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
    fig.savefig("results/d{}s{}t{}.pdf".format(dim, smooth, times))
    text_data.close()


def func(name, n, rec=0, nys=0):
    if name == 'N. + emp':
        pts_rec = gen_params(rec)
        pts_nys = gen_params(nys)
        idx, w = enys.recombination(
            pts_rec, pts_nys, n, use_obj=True, rand_SVD=False)
        x = pts_rec[idx]
        return x, w
    elif name == 'N. + emp + opt':
        pts_rec = gen_params(rec)
        pts_nys = gen_params(nys)
        idx, w = enys.recombination(
            pts_rec, pts_nys, n, use_obj=True, rand_SVD=False)
        x = pts_rec[idx]
        w = grlp.QP(k(x, x), k_exp(x), k_exp_exp(), nonnegative=True)
        return x, w
    elif name == 'M. + emp':
        pts_rec = gen_params(rec)
        idx, w = emer.recombination(pts_rec, n)
        x = pts_rec[idx]
        return x, w
    elif name == 'M. + emp + opt':
        pts_rec = gen_params(rec)
        idx, w = emer.recombination(pts_rec, n)
        x = pts_rec[idx]
        w = grlp.QP(k(x, x), k_exp(x), k_exp_exp(), nonnegative=True)
        return x, w
    elif name == 'Monte Carlo':
        return mc(n)
    elif name == 'iid Bayes':
        return mc_bayes(n)
    elif name == 'Uniform Grid':
        return ug_bayes(n)
    elif name == 'Halton':
        x = gen_params(1)
        _, d = x.shape
        sampler = Halton(d=d)
        x = sampler.random(n)
        w = np.ones(n)/n
        return x, w
    elif name == 'Halton + opt':
        x = gen_params(1)
        _, d = x.shape
        sampler = Halton(d=d)
        x = sampler.random(n)
        w = grlp.QP(k(x, x), k_exp(x), k_exp_exp())
        return x, w


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


def ug_bayes(m, nn=False):
    pt = np.array([[i / m] for i in range(m)])
    w = np.array([1 / m for _ in range(m)])
    return pt, w
    # return eval(pt, QP(pt, nn))
