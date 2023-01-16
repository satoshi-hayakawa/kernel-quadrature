import nys_mer as enys
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import functools
import time

import sys
import os

from scipy.stats.qmc import Halton

sys.path.append(os.path.abspath(".."))
import grlp  # noqa
import thin_pp  # noqa

d_global = 1
s_global = 1


def gen_params(n):
    return np.random.normal(0, 1, n).reshape(-1, 1)


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
    return np.exp(X ** 2) /   # edit here to change the kernel


def experiments(
    dim=1,
    smooth=1,
    times=20,
    seed=None,
    np_seed=None,
):
    np.random.seed(np_seed)
    global d_global
    d_global = dim
    global s_global
    s_global = smooth
    enys.k = functools.partial(k, s=smooth)
    # if dim == 1:
    #     emer.k = functools.partial(k, s=smooth)

    text_data = open("results/d{}s{}t{}.txt".format(
        dim, smooth, times), 'w', encoding='utf-8')
    print("seed = {}, np_seed = {}".format(seed, np_seed), file=text_data)

    fig = plt.figure()
    x_ex = [4, 8, 16, 32, 64]  # , 128]  # [5, 10, 15, 20, 30, 40, 50, 65, 80]
    # m_names = ['N. + emp', 'N. + emp + opt',
    #            'Monte Carlo', 'iid Bayes', 'Halton', 'Halton + opt', 'Thinning', 'Thin + opt']
    if dim == 1:
        m_names = ['N. + emp', 'Monte Carlo', 'N. + emp + mer',
                   'iid Bayes', 'Uniform Grid']
        # m_names = ['N. + emp', 'N. + emp + opt', 'Monte Carlo', 'N. + emp + mer', 'N. + emp + mer + opt',
        #            'iid Bayes', 'Uniform Grid']  # , 'Thinning', 'Thin + opt']
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
        pts_nys = CUE_seq_generator(nys)  # gen_params(nys)
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
    elif name == 'N. + emp + mer':
        pts_rec = gen_params(rec)
        # pts_nys = gen_params(nys)
        pts_nys = CUE_seq_generator(nys)
        idx, w = enys.recombination(
            pts_rec, pts_nys, n, use_obj=True, rand_SVD=False, mer_kz=True)
        x = pts_rec[idx]
        return x, w
    elif name == 'N. + emp + mer + opt':
        pts_rec = gen_params(rec)
        # pts_nys = gen_params(nys)
        pts_nys = CUE_seq_generator(nys)
        ##
        ##
        ##
        # この点をDPPで置き換える！！！！
        ##
        ##
        ##
        idx, w = enys.recombination(
            pts_rec, pts_nys, n, use_obj=True, rand_SVD=False, mer_kz=True)
        x = pts_rec[idx]
        w = grlp.QP(k(x, x), k_exp(x), k_exp_exp(), nonnegative=True)
        return x, w
    # elif name == 'M. + emp':
    #     pts_rec = gen_params(rec)
    #     idx, w = emer.recombination(pts_rec, n)
    #     x = pts_rec[idx]
    #     return x, w
    # elif name == 'M. + emp + opt':
    #     pts_rec = gen_params(rec)
    #     idx, w = emer.recombination(pts_rec, n)
    #     x = pts_rec[idx]
    #     w = grlp.QP(k(x, x), k_exp(x), k_exp_exp(), nonnegative=True)
    #     return x, w
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
    elif name == 'Thinning':
        return ktpp(n, rec)
    elif name == 'Thin + opt':
        x, _ = ktpp(n, rec)
        w = grlp.QP(k(x, x), k_exp(x), k_exp_exp(), nonnegative=True)
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


def ktpp(n, rec):
    lsize = int(np.floor(np.log2(rec/n) + 1e-5))
    idx = gen_params(n * int(2 ** lsize))
    X = idx
    coreset = thin_pp.main(X, lsize, s_global, name='sob')  # thin_pp
    m = len(coreset)
    return idx[coreset], np.ones(m) / m

# by Belhadj et al.


def CUE_seq_generator(N_):
    x1 = np.random.randn(N_, N_)
    x2 = np.random.randn(N_, N_)
    x = (x1+x2*1j)/np.sqrt(2)
    q, r = np.linalg.qr(x)
    r = np.diag(np.divide(np.diag(r), np.abs(np.diag(r))))
    u = np.dot(q, r)
    s, _ = np.linalg.eig(u)
    s_angle = (np.angle(s, deg=False)/np.pi + np.ones(N_))/2
    return s_angle.reshape(-1, 1)
