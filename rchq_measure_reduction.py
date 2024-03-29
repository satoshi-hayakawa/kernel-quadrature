import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import functools
import time

import grlp
import emp_nys as enys

# global
lam = 1
num = 0
data = 0
data_test = 0
data_t_out = 0
k_exp_ = 0
k_exp_exp_ = 0


def gen_params(n):
    return np.random.randint(num, size=n)


def preprocess(data_name):
    # read data
    global data, data_test, data_t_out
    if data_name == '3Dnet':
        data_read = np.loadtxt('data/3D_spatial_network.txt',
                               delimiter=',', usecols=(1, 2, 3))
    else:
        data_read = np.loadtxt(
            'data/Combined Cycle Power Plant Data Set.txt', delimiter=',')
    np.random.shuffle(data_read)
    global num
    num, dim = data_read.shape
    if data_name == '3Dnet':
        num = num // 10
    data = data_read[:num, :]
    for i in range(dim):
        data[:, i] = (data[:, i] - np.mean(data[:, i])) / np.std(data[:, i])
    data_test = data[:, dim-1:dim].reshape((num,))
    data_t_out = data_test * (data[:, 0:1] >= 0).reshape((num,))
    data_t_out = data_t_out * (data[:, 1:2] >= 0).reshape((num,))
    global lam
    lam = median_heuristics()
    k_exp_comp()
    k_exp_exp_comp()


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
    times=50,
    np_seed=None
):
    np.random.seed(np_seed)
    preprocess(data_name)
    enys.k = functools.partial(k, data_k=data, lam_k=lam, kernel=kernel)

    text_data = open("results_rchq/{}_{}_t{}.txt".format(
        data_name, kernel, times), 'w', encoding='utf-8')
    print("np_seed = {}".format(np_seed), file=text_data)

    fig = plt.figure()
    x_ex = [5, 10, 20, 40, 60, 80, 120, 160]
    m_names = ['N. + emp + opt', 'Nyström', 'Nyström + opt',
               'FNE', 'FNE + opt', 'iid Bayes']
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
                    N = 20*n
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
    fig.savefig("results_rchq/{}_{}_t{}.pdf".format(data_name, kernel, times))
    text_data.close()


def func(name, n, rec=0, nys=0):
    x = []
    w = []
    idx = []
    fails = 0
    if 'FNE' in name:
        pts_rec = gen_params(rec)
        pts_nys = gen_params(nys)
        idx, w = enys.recombination(
            pts_rec, pts_nys, n, use_obj=False, rand_SVD=True)
        x = pts_rec[idx]
    if 'N. + emp' in name:
        pts_rec = gen_params(rec)
        pts_nys = gen_params(nys)
        idx, w = enys.recombination(
            pts_rec, pts_nys, n, use_obj=True, rand_SVD=False)
        x = pts_rec[idx]
    elif name == 'iid Bayes':
        x, w = mc_bayes(n)
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

    if 'opt' in name:
        w = grlp.QP(k(x, x), k_exp(x), k_exp_exp(), nonnegative=True)

    return x, w, fails


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


def mc_bayes(m, nn=False):
    pt = gen_params(m)
    return pt, grlp.QP(k(pt, pt), k_exp(pt), k_exp_exp(), nonnegative=nn)


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
