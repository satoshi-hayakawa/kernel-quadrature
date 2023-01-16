# mred = measure reduction

import nys_mer as enys
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import functools
import time

import sys
import os

from dppy.finite_dpps import FiniteDPP

sys.path.append(os.path.abspath(".."))
import grlp  # noqa
import thin_pp  # noqa


# global
lam = 1
num = 0
data = 0
data_test = 0
data_t_out = 0
k_exp_ = 0
k_exp_exp_ = 0

k_exp_test_ = 0
k_exp_exp_test_ = 0


def gen_params(n):
    return np.random.randint(num, size=n)


def preprocess(data_name, data_split=False):
    # read data
    global data, data_test, data_t_out
    if data_name == '3Dnet':  # or 'PPlant'
        data_read = np.loadtxt('../data/3D_spatial_network.txt',
                               delimiter=',', usecols=(1, 2, 3))
    else:
        data_read = np.loadtxt(
            '../data/Combined Cycle Power Plant Data Set.txt', delimiter=',')
    np.random.shuffle(data_read)
    global num
    num, dim = data_read.shape
    print(num)
    if data_name == '3Dnet':
        num = num // 100
    elif data_split:
        num = num // 2
    data = data_read[:num, :].copy()

    d_m = [np.mean(data[:, i]) for i in range(dim)]
    d_s = [np.std(data[:, i]) for i in range(dim)]

    for i in range(dim):
        data[:, i] = (data[:, i] - d_m[i]) / d_s[i]
        #data[:, i] = (data[:, i] - np.mean(data[:, i])) / np.std(data[:, i])

    if data_split:
        data_test = data_read[:2*num, :].copy()
        for i in range(dim):
            data_test[:, i] = (data_test[:, i] - d_m[i]) / d_s[i]
        # adding the same normalization to the test data

    idx_sort = np.argsort(data[:, 0])
    data = data[idx_sort]

    #data_test = data[:, dim-1:dim].reshape((num,))
    #data_t_out = data_test * (data[:, 0:1] >= 0).reshape((num,))
    #data_t_out = data_t_out * (data[:, 1:2] >= 0).reshape((num,))
    global lam
    lam = median_heuristics()
    k_exp_comp()
    k_exp_exp_comp()
    if data_split:
        k_exp_comp_test()
        k_exp_exp_comp_test()


def k(x, y=0, diag=False, data_k=None, lam_k=None, kernel=None):
    # x, y: array of indices
    if lam_k is None:
        lam_k = lam
    if data_k is None:
        data_k = data
    if np.isscalar(x):
        x = np.array([x])
    if diag:
        return np.ones(len(x))
    if np.isscalar(y):
        y = np.array([y])
    K = euclidean_distances(data_k[x, :], data_k[y, :], squared=True)
    if kernel is None:
        kernel = 'Gaussian'
    if kernel == 'Gaussian':
        return np.exp(- K / (2 * lam))  # Gaussian
    else:
        return 1 / (1 + K / (2 * lam))  # rational quadratic


def experiments(
    data_name='3Dnet',
    kernel='Gaussian',
    times=20,
    np_seed=None,
    data_split=False
):
    np.random.seed(np_seed)
    preprocess(data_name, data_split)
    enys.k = functools.partial(k, data_k=data, lam_k=lam, kernel=kernel)

    text_data = open("results/{}_{}_t{}.txt".format(data_name, kernel, times), 'w', encoding='utf-8') if not data_split else open(
        "results/{}_{}_t{}_split.txt".format(data_name, kernel, times), 'w', encoding='utf-8')
    print("np_seed = {}".format(np_seed), file=text_data)

    fig = plt.figure()
    # , 128]  # [5, 10, 20, 40]  # , 60, 80, 120, 160]
    x_ex = [4, 8, 16, 32, 64]
    m_names = ['Monte Carlo', 'N. + emp', 'N. + emp + mer']
    # m_names = ['N. + emp', 'N. + emp + opt', 'N. + emp + mer + opt',
    #            'Monte Carlo', 'iid Bayes', 'Herding', 'Herd + opt', 'Thinning', 'Thin + opt']
    #m_names = ['Thinning', 'Monte Carlo']
    methods = len(m_names)
    results = [[] for i in range(methods)]
    results_up = [[] for i in range(methods)]
    results_low = [[] for i in range(methods)]
    m_marks = ['x', 'o', '^', 'v', '+', '>', '<', 'd', 's', '8']
    for n in x_ex:
        print("{} points. ".format(n), file=text_data)
        for i in range(methods):
            start_time = time.perf_counter()
            res = np.zeros(times)
            for j in range(times):
                points, weights = func(m_names[i], n, rec=n*n, nys=20*n)
                res[j] = eval(points, weights) if not data_split else eval_test(
                    points, weights)
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
    if not data_split:
        fig.savefig("results/{}_{}_t{}.pdf".format(data_name, kernel, times))
    else:
        fig.savefig(
            "results/{}_{}_t{}_split.pdf".format(data_name, kernel, times))
    text_data.close()


def func(name, n, rec=0, nys=0):
    if name == 'N. + emp':
        pts_rec = gen_params(rec)
        # pts_nys_ = gen_params(10 * nys)
        # pts_nys = pts_nys_[dpp_sampling(pts_nys_, nys)]
        pts_nys = weighted_sampling(nys)
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
        # pts_nys_ = gen_params(10 * nys)
        # pts_nys = pts_nys_[dpp_sampling(pts_nys_, nys)]
        pts_nys = weighted_sampling(nys)
        idx, w = enys.recombination(
            pts_rec, pts_nys, n, use_obj=True, rand_SVD=False, mer_kz=True)
        x = pts_rec[idx]
        return x, w
    elif name == 'N. + emp + mer + opt':
        pts_rec = gen_params(rec)
        pts_nys_ = gen_params(10 * nys)
        pts_nys = dpp_sampling(pts_nys_, nys)
        idx, w = enys.recombination(
            pts_rec, pts_nys, n, use_obj=True, rand_SVD=False, mer_kz=True)
        x = pts_rec[idx]
        w = grlp.QP(k(x, x), k_exp(x), k_exp_exp(), nonnegative=True)
        return x, w
    elif name == 'Monte Carlo':
        return mc(n)
    elif name == 'iid Bayes':
        return mc_bayes(n)
    elif name == 'Herding':
        return herding(n)
    elif name == 'Herd + opt':
        x, _ = herding(n)
        w = grlp.QP(k(x, x), k_exp(x), k_exp_exp(), nonnegative=True)
        return x, w
    elif name == 'Thinning':
        return ktpp(n, rec)
    elif name == 'Thin + opt':
        x, _ = ktpp(n, rec)
        w = grlp.QP(k(x, x), k_exp(x), k_exp_exp(), nonnegative=True)
        return x, w
    else:
        print("There is not such a method")


def k_exp_exp():
    return k_exp_exp_


def k_exp_exp_comp():  # post computation
    global k_exp_exp_
    k_exp_exp_ = np.sum(k_exp_) / num


def k_exp(x):
    return k_exp_[x]


def k_exp_comp():  # post computation
    r = np.ones((num,))
    r /= num
    xal = np.arange(num)
    xsp = np.array_split(xal, np.minimum(50, len(xal)))
    dots = [k(x, xal) @ r for x in xsp]
    global k_exp_
    k_exp_ = np.concatenate(dots)


def median_heuristics():
    num_mh = np.minimum(10000, num)
    xal = np.arange(num_mh)
    xsp = np.array_split(xal, np.minimum(50, len(xal)))
    tmp = np.zeros(num_mh)
    for x in xsp:
        tmp = np.append(tmp, euclidean_distances(
            data[x, :], data[xal, :], squared=True).reshape(num_mh * len(x)))
    return np.median(tmp) / 2


def eval(x, w, pr=False):
    if pr == True:
        print(w)
    if len(x) == 0:
        return 10000000000
    m = len(x)
    tmp = np.transpose(w) @ k_exp(x)
    ret = (k_exp_exp() - tmp) + (np.transpose(w) @ k(x, x) @ w - tmp)
    return ret


def mc(m):
    pt = gen_params(m)
    w = np.ones(m)/m
    return pt, w


def mc_bayes(m, nn=False):
    pt = gen_params(m)
    return pt, grlp.QP(k(pt, pt), k_exp(pt), k_exp_exp(), nonnegative=nn)


def herding(m, reweight=False):
    ip = np.zeros(num)
    xal = np.arange(num)
    xnew = np.random.randint(num, size=1)
    x = [xnew]
    for i in range(m - 1):
        ip = ip + k(xal, xnew).reshape((num,)) - k_exp(xal)
        xnew = np.argmin(ip)
        x = np.append(x, xnew)
    x = x.astype(int)
    if reweight:
        return x, grlp.QP(k(x, x), k_exp(x), k_exp_exp(), nonnegative=True)
    else:
        return x, np.ones(m) / m


def ktpp(n, rec):
    lsize = int(np.floor(np.log2(rec/n) + 1e-5))
    idx = gen_params(n * int(2 ** lsize))
    X = data[idx]
    coreset = thin_pp.main(X, lsize, lam)  # thin_pp
    m = len(coreset)
    return idx[coreset], np.ones(m) / m

# below is for the MMD comparison with another set of empirical data
# using copy & paste to minimize the change to the original version of the code..


def k_exp_exp_test():
    return k_exp_exp_test_


def k_exp_exp_comp_test():  # post computation
    global k_exp_exp_test_
    k_exp_exp_test_ = np.sum(k_exp_test_[-num:]) / num


def k_exp_test(x):
    return k_exp_test_[x]


def k_exp_comp_test():  # post computation
    r = np.ones((num,))
    r /= num
    x_test = np.arange(num, 2*num)
    xal = np.arange(2*num)
    xsp = np.array_split(xal, np.minimum(50, len(xal)))
    dots = [k(x, x_test, data_k=data_test) @ r for x in xsp]
    global k_exp_test_
    k_exp_test_ = np.concatenate(dots)


def eval_test(x, w, pr=False):
    if pr == True:
        print(w)
    if len(x) == 0:
        return 10000000000
    m = len(x)
    tmp = np.transpose(w) @ k_exp_test(x)
    ret = (k_exp_exp_test() - tmp) + (np.transpose(w) @ k(x, x) @ w - tmp)
    return ret


def dpp_sampling(x, l):  # l-point DPP from x
    gmat = k(x, x)
    svd = TruncatedSVD(n_components=l)
    svd.fit(gmat)
    eig_vals = np.ones(l)
    eig_vecs = svd.components_.T
    # print(eig_vecs.shape)
    # print(eig_vecs.T @ eig_vecs)
    dpp = FiniteDPP(kernel_type='correlation',
                    projection=True,
                    **{'K_eig_dec': (eig_vals, eig_vecs)})
    rng = np.random.RandomState(0)
    return dpp.sample_exact(mode='GS', random_state=rng)


def weighted_sampling(m):
    before = np.random.randint(num * num, size=m)
    after = np.int_(np.floor(np.sqrt(before + 1e-5)))
    return after
