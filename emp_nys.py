import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import euclidean_distances


def k(x, y=0, diag=False):
    lam = 10
    if np.isscalar(x):
        x = np.array([x])
    if np.isscalar(y):
        y = np.array([y])
    if diag:
        return np.ones(len(x))
    else:
        K = euclidean_distances(x, y, squared=True)
        return np.exp(- K / (2 * lam))  # Gaussian
        # return 1 / (1 + K / (2 * lam)) # rational quadratic


def recombination(
    pts_rec,  # random sample for recombination
    pts_nys,  # number of points used for approximating kernel
    num_pts,  # number of points finally returned
    init_weights=0,  # initial weights of the sample for recombination
    use_obj=False,  # whether or not using objective
    rand_SVD=True  # whether or not using randomized SVD
):
    return rc_kernel_svd(pts_rec, pts_nys, num_pts, mu=init_weights, use_obj=use_obj, rand_SVD=rand_SVD)


def ker_svd_sparsify(pt, s, rand_SVD=True):
    svd = TruncatedSVD(n_components=s) if rand_SVD else TruncatedSVD(
        n_components=s, algorithm='arpack')
    svd.fit(k(pt, pt))
    return svd.singular_values_, svd.components_
    # ul, dia, ur = np.linalg.svd(k(pt, pt), hermitian=True)
    # return ur[:s, :]


def rc_kernel_svd(samp, pt, s, mu=0, use_obj=True, rand_SVD=True):
    # Nystrom
    svs, U = ker_svd_sparsify(pt, s - 1, rand_SVD)
    obj = 0
    if use_obj:
        idx_feasible = svs >= 1e-10
        inv_svs = np.zeros(len(svs))
        inv_svs[idx_feasible] = np.sqrt(1/svs[idx_feasible])
        sur_svs = np.reshape(inv_svs, (-1, 1))
        obj = k(samp, diag=True)
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

    w_star, idx_star = Mod_Tchernychova_Lyons(
        samp, U, pt, obj, mu, use_obj=use_obj)

    if use_obj:
        # final sparsification
        Xp = U @ k(pt, samp[idx_star])
        Xp = np.append(Xp, np.ones((1, len(idx_star))), axis=0)
        _, _, w_null = np.linalg.svd(Xp)
        w_null = w_null[-1]
        if np.dot(obj[idx_star], w_null) < 0:
            w_null = -w_null

        lm = len(w_star)
        plis = w_null > 0
        alpha = np.zeros(lm)
        alpha[plis] = w_star[plis] / w_null[plis]
        idx = np.arange(lm)[plis]
        idx = idx[np.argmin(alpha[plis])]
        w_star = w_star-alpha[idx]*w_null
        w_star[idx] = 0.

        idx_ret = idx_star[w_star > 0]
        w_ret = w_star[w_star > 0]
        return idx_ret, w_ret

    else:
        return idx_star, w_star

# Mod_Tchernychova_Lyons is modification of Tcherynychova_Lyons from https://github.com/FraCose/Recombination_Random_Algos/blob/master/recombination.py


def Mod_Tchernychova_Lyons(samp, U_svd, pt_nys, obj=0, mu=0, use_obj=True, DEBUG=False):

    N = len(samp)
    n, l = U_svd.shape
    if use_obj:
        n = n + 1

    # tic = timeit.default_timer()

    number_of_sets = 2*(n+1)

    if np.all(obj == 0) or len(obj) != N:
        obj = np.zeros(N)
    if np.all(mu == 0) or len(mu) != N or np.any(mu < 0):
        mu = np.ones(N)/N

    idx_story = np.arange(N)
    idx_story = idx_story[mu != 0]
    remaining_points = len(idx_story)

    while True:

        if remaining_points <= n+1:
            idx_star = np.arange(len(mu))[mu > 0]
            w_star = mu[idx_star]
            # toc = timeit.default_timer()-tic
            return w_star, idx_star
            # return w_star, idx_star, X[idx_star], toc, ERR, np.nan, np.nan # original
        elif n+1 < remaining_points <= number_of_sets:
            X_mat = U_svd @ k(pt_nys, samp[idx_story])
            if use_obj:
                X_mat = np.append(X_mat, np.reshape(
                    obj[idx_story], (1, -1)), axis=0)
            w_star, idx_star, x_star, _, ERR, _, _ = Tchernychova_Lyons_CAR(
                np.transpose(X_mat), np.copy(mu[idx_story]), DEBUG)
            idx_story = idx_story[idx_star]
            mu[:] = 0.
            mu[idx_story] = w_star
            idx_star = idx_story
            w_star = mu[mu > 0]
            # toc = timeit.default_timer()-tic
            return w_star, idx_star
            # return w_star, idx_star, x_star, toc, ERR, np.nan, np.nan

        # remaining points at the next step are = remaining_points/card*(n+1)

        # number of elements per set
        number_of_el = int(remaining_points/number_of_sets)
        # WHAT IF NUMBER OF EL == 0??????
        # IT SHOULD NOT GET TO THIS POINT GIVEN THAT AT THE END THERE IS A IF

        # X_tmp = np.zeros((number_of_sets, n))
        # mu_tmp = np.empty(number_of_sets)

        idx = idx_story[:number_of_el*number_of_sets].reshape(number_of_el, -1)
        X_for_nys = np.zeros((l, number_of_sets))
        X_for_obj = np.zeros((1, number_of_sets))
        for i in range(number_of_el):
            idx_tmp_i = idx_story[i * number_of_sets:(i+1)*number_of_sets]
            X_for_nys += np.multiply(k(pt_nys,
                                     samp[idx_tmp_i]), mu[np.newaxis, idx_tmp_i])
            if use_obj:
                X_for_obj += np.multiply(np.reshape(
                    obj[idx_tmp_i], (1, -1)), mu[np.newaxis, idx_tmp_i])
        # for i in range(number_of_el):
        #     X_mat = U_svd @ k(pt_nys, samp[idx_story[i *
        #                                              number_of_sets:(i+1)*number_of_sets]])
        #     if use_obj:
        #         X_mat = np.append(X_mat, np.reshape(
        #             obj[idx_story[i*number_of_sets:(i+1)*number_of_sets]], (1, -1)), axis=0)
        #     X_tmp += np.multiply(np.transpose(X_mat), mu[idx_story[i *
        #                          number_of_sets:(i+1)*number_of_sets], np.newaxis])
        X_tmp_tr = U_svd @ X_for_nys
        if use_obj:
            X_tmp_tr = np.append(X_tmp_tr, X_for_obj, axis=0)
        X_tmp = np.transpose(X_tmp_tr)

        tot_weights = np.sum(mu[idx], 0)

        idx_last_part = idx_story[number_of_el*number_of_sets:]

        if len(idx_last_part):
            X_mat = U_svd @ k(pt_nys, samp[idx_last_part])
            if use_obj:
                X_mat = np.append(X_mat, np.reshape(
                    obj[idx_last_part], (1, -1)), axis=0)
            X_tmp[-1] += np.multiply(np.transpose(X_mat),
                                     mu[idx_last_part, np.newaxis]).sum(axis=0)
            tot_weights[-1] += np.sum(mu[idx_last_part], 0)

        X_tmp = np.divide(X_tmp, tot_weights[np.newaxis].T)

        w_star, idx_star, _, _, ERR, _, _ = Tchernychova_Lyons_CAR(
            X_tmp, np.copy(tot_weights))

        idx_tomaintain = idx[:, idx_star].reshape(-1)
        idx_tocancel = np.ones(idx.shape[1]).astype(bool)
        idx_tocancel[idx_star] = 0
        idx_tocancel = idx[:, idx_tocancel].reshape(-1)

        mu[idx_tocancel] = 0.
        mu_tmp = np.multiply(mu[idx[:, idx_star]], w_star)
        mu_tmp = np.divide(mu_tmp, tot_weights[idx_star])
        mu[idx_tomaintain] = mu_tmp.reshape(-1)

        idx_tmp = idx_star == number_of_sets-1
        idx_tmp = np.arange(len(idx_tmp))[idx_tmp != 0]
        # if idx_star contains the last barycenter, whose set could have more points
        if len(idx_tmp) > 0:
            mu_tmp = np.multiply(mu[idx_last_part], w_star[idx_tmp])
            mu_tmp = np.divide(mu_tmp, tot_weights[idx_star[idx_tmp]])
            mu[idx_last_part] = mu_tmp
            idx_tomaintain = np.append(idx_tomaintain, idx_last_part)
        else:
            idx_tocancel = np.append(idx_tocancel, idx_last_part)
            mu[idx_last_part] = 0.

        idx_story = np.copy(idx_tomaintain)
        remaining_points = len(idx_story)
        # remaining_points = np.sum(mu>0)


# Tchernychova_Lyons_CAR is taken from https://github.com/FraCose/Recombination_Random_Algos/blob/master/recombination.py


def Tchernychova_Lyons_CAR(X, mu, DEBUG=False):
    # this functions reduce X from N points to n+1

    # com = np.sum(np.multiply(X,mu[np.newaxis].T),0)
    X = np.insert(X, 0, 1., axis=1)
    N, n = X.shape
    U, Sigma, V = np.linalg.svd(X.T)
    # np.allclose(U @ np.diag(Sigma) @ V, X.T)
    U = np.append(U, np.zeros((n, N-n)), 1)
    Sigma = np.append(Sigma, np.zeros(N-n))
    Phi = V[-(N-n):, :].T
    cancelled = np.array([], dtype=int)

    for _ in range(N-n):

        # alpha = mu/Phi[:, 0]
        lm = len(mu)
        plis = Phi[:, 0] > 0
        alpha = np.zeros(lm)
        alpha[plis] = mu[plis] / Phi[plis, 0]
        idx = np.arange(lm)[plis]
        idx = idx[np.argmin(alpha[plis])]
        cancelled = np.append(cancelled, idx)
        mu[:] = mu-alpha[idx]*Phi[:, 0]
        mu[idx] = 0.

        if DEBUG and (not np.allclose(np.sum(mu), 1.)):
            # print("ERROR")
            print("sum ", np.sum(mu))

        Phi_tmp = Phi[:, 0]
        Phi = np.delete(Phi, 0, axis=1)
        Phi = Phi - np.matmul(Phi[idx, np.newaxis].T,
                              Phi_tmp[:, np.newaxis].T).T/Phi_tmp[idx]
        Phi[idx, :] = 0.

    w_star = mu[mu > 0]
    idx_star = np.arange(N)[mu > 0]
    return w_star, idx_star, np.nan, np.nan, 0., np.nan, np.nan
