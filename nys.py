import math
import random
import numpy as np
import gurobipy as gp
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

eig = True

def main():

	failed = 0

	random.seed()

	fig = plt.figure()

	rc, rc_ker_svd, mc_error, iid_bayes_error, ug_error = ([], [], [], [], [])
	times = 50
	x_ex = np.array([5, 10, 20, 40, 80])
	for m in x_ex:
		n = 10 * m
		res1, res2, res3, res4, res5 = (0, 0, 0, 0, 0)
		for tm in range(times):
			tmp1 = rc_eval(m, n)
			while tmp1 > 1000000000:
				failed += 1
				tmp1 = rc_eval(m, n)
			res1 += tmp1 / times # RC

			tmp2 = rc_kernel_svd_eval(n, m, n)
			while tmp2 > 1000000000:
				failed += 1
				tmp2 = rc_kernel_svd_eval(n, m, n)
			res2 += tmp2 / times # RC with kernel + SVD

			res3 += mc(m) / times # Monte Carlo
			res4 += mc_bayes(m) / times # iid Bayesian
			res5 += ug_bayes(m) / times # uniform grid
		rc += [math.log10(res1)]
		rc_ker_svd += [math.log10(res2)]
		mc_error += [math.log10(res3)]
		iid_bayes_error += [math.log10(res4)]
		ug_error += [math.log10(res5)]

	x = np.log10(x_ex)

	plt.plot(x, rc, label = 'Mercer', marker = 'o')
	plt.plot(x, rc_ker_svd, label = 'Nyström', marker = 'v')
	plt.plot(x, mc_error, label = 'Monte Carlo', marker = '^')
	plt.plot(x, iid_bayes_error, label = 'iid Bayes', marker = 's')
	plt.plot(x, ug_error, label = 'Uniform Grid', marker = '*')

	plt.legend(loc = 'lower left')
	plt.xlabel("$\mathrm{log}_{10} n$")
	plt.ylabel("$\mathrm{log}_{10} (\mathrm{wce})^2$")

	plt.show()
	fig.savefig("img.pdf")

	print(failed)

def k(x, y):
	if np.isscalar(x):
		x = np.array([x])
	if np.isscalar(y):
		y = np.array([y])
	m = len(x)
	n = len(y)
	X = np.zeros((m, n))
	for i in range(n):
		X[:, i] += x
	for j in range(m):
		X[j, :] -= np.transpose(y)
	return sob(X, 1) ## edit here to change the kernel


def LP(K, B, set_objective = False, obj = []):

	m, n = K.shape

	model = gp.Model("test")
	x = model.addMVar(n)
	model.update()

	for i in range(m):
		model.addConstr(K[i, :] @ x == B[i])
	print("done")
	model.addConstr(x >= 0)

	if(set_objective == True):
		model.setObjective(obj @ x)
	else:
		model.setObjective(0)
	model.optimize()

	sol = []
	if model.Status == gp.GRB.OPTIMAL:
		for i in range(n):
			if x[i].X > 0:
				sol += [(i, x[i].X)]
	else:
		print("FAILED")

	return sol


def QP(pts, nonnegative=False):
	return QP_(k(pts, pts), k_exp(pts), nonnegative)


def QP_(A, EA, nonnegative=False):

	m = len(EA)

	model = gp.Model("test")
	w = model.addMVar(m)
	model.update()

	if nonnegative == True:
		model.addConstr(w >= 0)

	model.setObjective(k_exp_exp() + w @ A @ w - 2 * EA @ w)
	model.optimize()

	wei = []
	if model.Status == gp.GRB.OPTIMAL:
		for i in range(m):
			wei += [w[i].X]
	else:
		print("FAILED")

	return wei

def eval(x, w, pr=False):
	if pr == True:
		print(w)
	if len(x) == 0:
		return 10000000000
	m = len(x)
	ret = k_exp_exp()
	ret -= 2 * np.transpose(w) @ k_exp(x)
	ret += np.transpose(w) @ k(x, x) @ w
	return ret[0, 0]


def k_exp(x):
	if np.isscalar(x):
		return 1
	else:
		return np.ones((len(x),))

def k_exp_exp():
	return np.ones((1, 1))

def sob(y, s):
	x = np.maximum(0, y) + (1 + np.minimum(0, y)) * (y < 0)
	if s == 1:
		return 1 + 2 * (math.pi ** 2) * ((x ** 2) - x + 1 / 6)
	if s == 2:
		return 1 - (math.pi ** 4) * 2 / 3 * ((x ** 4) - 2 * (x ** 3) + (x ** 2) - 1 / 30)
	if s == 3:
		return 1 + (math.pi ** 6) * 4 / 45 * ((x ** 6) - 3 * (x ** 5) + 5 / 2 * (x ** 4) - (x ** 2) / 2 + 1 / 42)

def mc(m):
	pt = np.zeros(m)
	for i in range(m):
		pt[i] = random.random()
	ret = k_exp_exp() # Monte Carlo
	ret -= 2 / m * np.sum(k_exp(pt))
	ret += 1 / (m ** 2) * np.sum(k(pt, pt))
	return ret[0, 0]

def mc_bayes(m, nn=False):
	pt = np.zeros(m)
	for i in range(m):
		pt[i] = random.random()
	return eval(pt, QP(pt, nn))

def ug_bayes(m, nn=False):
	pt = np.array([i / m for i in range(m)])
	w = np.array([1 / m for i in range(m)])
	return eval(pt, w)
	# return eval(pt, QP(pt, nn))

def rc_trig_uniform(m, n):

	samp = np.zeros((n,))
	B = np.zeros(m)
	B[m - 1] = 1
	K = np.ones((m, n))

	for j in range(n):
		samp[j] = random.random()

	a = np.array([i % 2 for i in range(m - 1)])
	b = np.array([2 * (i // 2) + 2 for i in range(m - 1)])

	K_cos = np.cos(math.pi * b[:, None] @ samp[None, :])
	K_sin = np.sin(math.pi * b[:, None] @ samp[None, :])
	mask = a == 0
	K[:m-1, :] = K_cos * mask[:, None] + K_sin * (1 - mask[:, None])

	ob = np.zeros(n)
	for j in range(n):
		ob[j] = k(samp[j], samp[j])
	sol = LP(K, B, True, ob)

	points = []
	weights = []
	for a, w_a in sol:
		points += [samp[a]]
		weights += [w_a]

	return points, weights

def rc_eval(m, n):
	points, weights = rc_trig_uniform(m, n)
	rc = eval(points, weights)
	# rc_qp = eval(points, QP(points, True))
	return rc

def rc_kernel(m, n):
	samp = np.zeros(n)
	pt = np.zeros(m - 1)
	B = np.ones((m,))
	K = np.ones((m, n))

	for i in range(m - 1):
		pt[i] = random.random()
	for j in range(n):
		samp[j] = random.random()

	B[:m-1] = k_exp(pt)
	K[:m-1, :] = k(pt, samp)

	sol = LP(K, B)

	points = []
	weights = []
	for a, w_a in sol:
		points += [samp[a]]
		weights += [w_a]

	return points, weights

def rc_kernel_eval(m, n):
	points, weights = rc_kernel(m, n)
	rc_ker = eval(points, weights)
	# rc_ker_qp = eval(points, QP(points, True))
	return rc_ker

def ker_svd_sparsify(pt, s):
	svd = TruncatedSVD(n_components = s, algorithm = 'arpack') #, algorithm = 'arpack'
	svd.fit(k(pt, pt))
	return svd.components_
	#ul, dia, ur = np.linalg.svd(k(pt, pt), hermitian=True)
	#return ur[:s, :]

def rc_kernel_svd(t, s, n):
	samp = np.zeros(n)
	pt = np.zeros(t)
	B = np.ones(s,)
	K = np.ones((s, n))

	for i in range(t):
		pt[i] = random.random()
	for j in range(n):
		samp[j] = random.random()

	U = ker_svd_sparsify(pt, s - 1)

	B[:s-1] = U @ k_exp(pt)
	K[:s-1, :] = U @ k(pt, samp)

	sol = LP(K, B)

	points = []
	weights = []
	for a, w_a in sol:
		points += [samp[a]]
		weights += [w_a]

	return points, weights

def rc_kernel_svd_eval(t, s, n):
	points, weights = rc_kernel_svd(t, s, n)
	return eval(points, weights)

if __name__ == "__main__":
	# m = int(input())
	main()










