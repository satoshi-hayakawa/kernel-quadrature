import gurobipy as gp


def LP(K, B, set_objective=False, obj=[]):
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


def QP(A, EA, EE=0, nonnegative=False):
    m = len(EA)
    model = gp.Model("test")
    w = model.addMVar(m)
    model.update()
    if nonnegative == True:
        model.addConstr(w >= 0)
    model.setObjective(w @ A @ w - 2 * EA @ w + EE)
    model.optimize()
    wei = []
    if model.Status == gp.GRB.OPTIMAL:
        for i in range(m):
            wei += [w[i].X]
    else:
        print("FAILED")
    return wei
