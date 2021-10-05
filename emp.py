import math
import random
import numpy as np
import gurobipy as gp
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt

# global 
lam = 1
num = 0
data = 0
data_test = 0
data_t_out = 0
k_exp = 0
k_exp_exp = 0

def preprocess():
	# read data
	global data, data_test, data_t_out
	data_read = np.loadtxt('3D_spatial_network.txt', delimiter = ',', usecols = (1, 2, 3))
	#data_read = np.loadtxt('Combined Cycle Power Plant Data Set.txt', delimiter = ',')
	np.random.shuffle(data_read)
	global num
	num, dim = data_read.shape
	num = num // 10
	data = data_read[:num,:]
	for i in range(dim):
		data[:, i] = (data[:, i] - np.mean(data[:, i])) / np.std(data[:, i])
	data_test = data[:, dim-1:dim].reshape((num,))
	data_t_out = data_test * (data[:,0:1] >= 0).reshape((num,))
	data_t_out = data_t_out * (data[:,1:2] >= 0).reshape((num,))
	global lam
	lam = median_heuristics()
	k_exp_()
	k_exp_exp_()

def k(x, y):
	# x, y: array of indices
	if np.isscalar(x):
		x = np.array([x])
	if np.isscalar(y):
		y = np.array([y])
	K = euclidean_distances(data[x, :], data[y, :], squared = True)
	#return np.exp(- K / (2 * lam)) # Gaussian
	return 1 / (1 + K / (2 * lam)) # rational quadratic


def main():
	np.random.seed()
	preprocess()
	fig, (L, M, R) = plt.subplots(ncols = 3, sharex = True, figsize = (15, 4))
	methods = 5
	results = [[], [], [], [], []]
	test_res = [[], [], [], [], []]
	test_out_res = [[], [], [], [], []]
	m_names = ['N-reweight', 'Nyström', 'Monte Carlo', 'iid Bayes', 'herding']
	m_marks = ['*', 'v', '^', 's', 'o']
	times = 1
	x_ex = np.array([5, 10, 20, 40, 80, 120, 160])
	failed = 0
	test_exp = np.mean(data_test)
	t_out_exp = np.mean(data_t_out)
	for m in x_ex:
		n = 20 * m
		res = np.zeros(methods)
		t_res = np.zeros(methods)
		t_out_res = np.zeros(methods)
		for i in range(methods):
			for count_times in range(times):
				points, weights = func(m, n, i)
				while len(points) == 0:
					failed += 1
					points, weights = func(m, n, i)
				res[i] += eval(points, weights) / times
				t_res[i] += eval_test(points, weights, test_exp, data_test) / times
				t_out_res[i] += eval_test(points, weights, t_out_exp, data_t_out) / times
			results[i] += [math.log10(res[i])]
			test_res[i] += [math.log10(t_res[i])]
			test_out_res[i] += [math.log10(t_out_res[i])]

	x = np.log10(x_ex)
	# x = x_ex

	for i in range(methods):
		L.plot(x, results[i], label = m_names[i], marker = m_marks[i])
		M.plot(x, test_res[i], label = m_names[i], marker = m_marks[i])
		R.plot(x, test_out_res[i], label = m_names[i], marker = m_marks[i])
	L.legend(loc = 'lower left')
	L.set_xlabel("$\mathrm{log}_{10} n$")
	M.set_xlabel("$\mathrm{log}_{10} n$")
	R.set_xlabel("$\mathrm{log}_{10} n$")
	L.set_ylabel("$\mathrm{log}_{10} (\mathrm{wce})^2$")
	M.set_ylabel("$\mathrm{log}_{10} (\mathrm{error})^2$")
	R.set_ylabel("$\mathrm{log}_{10} (\mathrm{error})^2$")
	plt.tight_layout()
	plt.show()
	fig.savefig("img.pdf")
	print(failed)
	print(lam)
	print(num)

def func(m, n, i):
	if i == 0:
		x, w = rc_kernel_svd(n, m, n)
		if len(x) > 0:
			return x, QP(x, nonnegative = True)
		else:
			return [], []
	elif i == 1:
		return rc_kernel_svd(n, m, n)
	elif i == 2:
		return mc(m)
	elif i == 3:
		return mc_bayes(m)
	else:
		return herding(m)

def k_exp_exp_(): # post computation
	global k_exp_exp
	k_exp_exp = np.sum(k_exp) / num

def k_exp_(): # post computation
	r = np.ones((num,))
	r /= num
	xal = np.arange(num)
	xsp = np.array_split(xal, 100)
	dots =[k(x, xal) @ r for x in xsp]
	global k_exp
	k_exp = np.concatenate(dots)

def median_heuristics():
	num = 10000
	xal = np.arange(num)
	xsp = np.array_split(xal, 100)
	tmp = np.zeros(num)
	for x in xsp:
		tmp = np.append(tmp, euclidean_distances(data[x, :], data[xal, :], squared = True).reshape(num * len(x)))
	return np.median(tmp) / 2

def eval(x, w, pr=False):
	if pr == True:
		print(w)
	if len(x) == 0:
		return np.nan
	m = len(x)
	ret = k_exp_exp
	ret = ret - 2 * np.transpose(w) @ k_exp[x]
	ret = ret + np.transpose(w) @ k(x, x) @ w
	return ret

def eval_test(x, w, ans, arr):
	return (ans - np.transpose(w) @ arr[x]) ** 2

def mc(m):
	pt = np.random.randint(num, size = m)
	wei = np.ones(m) / m
	return pt, wei

def mc_bayes(m, nn=False):
	pt = np.random.randint(num, size = m)
	return pt, QP(pt, nn)

def ker_svd_sparsify(pt, s):
	svd = TruncatedSVD(n_components = s) #, algorithm = 'arpack'
	svd.fit(k(pt, pt))
	return svd.singular_values_, svd.components_
	#ul, dia, ur = np.linalg.svd(k(pt, pt), hermitian=True)
	#return ur[:s, :]

def rc_kernel_svd(t, s, n, set_objective = False):
	samp = np.random.randint(num, size = n)
	pt = np.random.randint(num, size = t)
	B = np.ones(s,)
	K = np.ones((s, n))

	svs, U = ker_svd_sparsify(pt, s - 1)

	B[:s-1] = U @ k_exp[pt]
	K[:s-1, :] = U @ k(pt, samp)

	obj = - (svs[None,:] ** 2) @ (K[:s-1, :] ** 2)
	obj = np.reshape(obj, n)

	if set_objective:
		sol = LP(K, B, set_objective = True, obj = obj)
	else:
		sol = LP(K, B)

	points = []
	weights = []
	for a, w_a in sol:
		points += [samp[a]]
		weights += [w_a]
	return points, weights

def herding(m, reweight = False):
	ip = np.zeros(num)
	xal = np.arange(num)
	xnew = np.random.randint(num, size = 1)
	x = [xnew]
	for i in range(m - 1):
		ip = ip + k(xal, xnew).reshape((num,)) - k_exp
		xnew = np.argmin(ip)
		x = np.append(x, xnew)
	x = x.astype(int)
	if reweight:
		return x, QP(x, nonnegative = True)
	else:
		return x, np.ones(m) / m

def LP(K, B, set_objective = False, obj = []):
	m, n = K.shape
	model = gp.Model("test")
	x = model.addMVar(n)
	model.update()
	for i in range(m):
		model.addConstr(K[i, :] @ x == B[i])
	# print("done")
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
	return QP_(k(pts, pts), k_exp[pts], nonnegative)


def QP_(A, EA, nonnegative=False):
	m = len(EA)
	model = gp.Model("test")
	# model.params.NonConvex = 2
	w = model.addMVar(m)
	model.update()
	if nonnegative == True:
		model.addConstr(w >= 0)
	model.setObjective(k_exp_exp + w @ A @ w - 2 * EA @ w)
	model.optimize()
	wei = []
	if model.Status == gp.GRB.OPTIMAL:
		for i in range(m):
			wei += [w[i].X]
	else:
		print("FAILED")
	return wei

if __name__ == "__main__":
	main()










