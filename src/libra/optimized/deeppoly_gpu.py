import numpy as np
import cupy as cp
from numba import cuda
from libra.optimized.commons import texpr_to_dict,get_bounds_single,ineq_str
from libra.core.cfg import Node, Function, Activation
import copy

def back_propagate_l1_GPU(self,ln_coeff, ineq_prev_lte,
                          ineq_prev_gte):
    @cuda.jit
    def back_propagate_helper(l1_lte, l1_gte, coeff, ineq_prev_lte_i, ineq_prev_gte_i):
        k, p = cuda.grid(2)
        if (k >= len(coeff) or k >= len(l1_lte) or k >= len(l1_gte) or p >= len(ineq_prev_lte_i) or p >= len(
                ineq_prev_gte_i)):
            return
        if (coeff[k] > 0):  # add check if k and p are valid
            l1_lte[k][p] += coeff[k] * ineq_prev_lte_i[p]  # should it be i or i-1?
            l1_gte[k][p] += coeff[k] * ineq_prev_gte_i[p]
        else:
            l1_lte[k][p] += coeff[k] * ineq_prev_gte_i[p]
            l1_gte[k][p] += coeff[k] * ineq_prev_lte_i[p]

    l1_lte = np.zeros((len(ln_coeff), len(ineq_prev_lte[1])))
    l1_gte = np.zeros((len(ln_coeff), len(ineq_prev_lte[1])))
    l1_lte[:, 0] = ln_coeff[:, 0]
    l1_gte[:, 0] = ln_coeff[:, 0]
    d_l1_lte = cp.array(l1_lte)
    d_l1_gte = cp.array(l1_gte)

    cuda_iters = (len(ln_coeff), len(ineq_prev_lte[1]))
    tpb = (min(32, cuda_iters[0]), min(32, cuda_iters[1]))
    bpg = (int(np.ceil(cuda_iters[0] / tpb[0])), int(np.ceil(cuda_iters[1] / tpb[1])))
    #print(f"tpb = {tpb}, bpg = {bpg}")
    for i in range(1, len(d_l1_lte)):
        d_coeff = cp.array(ln_coeff[:, i])

        d_ineq_prev_lte_i = cp.asarray(ineq_prev_lte[i])  # only load whats needed now to prevent out of memory

        d_ineq_prev_gte_i = cp.asarray(ineq_prev_gte[i])

        back_propagate_helper[bpg, tpb](d_l1_lte,
                                        d_l1_gte,
                                        d_coeff,
                                        d_ineq_prev_lte_i,
                                        d_ineq_prev_gte_i, )
        # pycuda.driver.Context.synchronize();
        l1_gte = cp.asnumpy(d_l1_gte)
        l1_lte = cp.asnumpy(d_l1_lte)
        return l1_lte, l1_gte

def get_bounds_GPU(d_l1_lte, d_l1_gte, l1_lb=-1, l1_ub=1):
    @cuda.jit
    def bound_helper(l1_lte, l1_gte, l1_lb, l1_ub, lbs, ubs):
        id = cuda.grid(1)
        if (id >= len(lbs)):
            return
        lbs[id] = l1_lte[id][0]
        for coeff in l1_lte[id, 1:]:
            if (coeff < 0):
                lbs[id] += coeff * l1_ub[0]
            else:
                lbs[id] += coeff * l1_lb[0]
        ubs[id] = l1_gte[id][0]
        for coeff in l1_gte[id, 1:]:
            if (coeff > 0):
                ubs[id] += coeff * l1_ub[0]
            else:
                ubs[id] += coeff * l1_lb[0]

    d_l1_lb = cp.asarray([l1_lb])
    d_l1_ub = cp.asarray([l1_ub])
    d_lbs = cp.zeros(d_l1_lte.shape[0])
    d_ubs = cp.zeros(d_l1_lte.shape[0])
    tpb = (min(1024, len(d_l1_lte)),)
    bpg = (int(np.ceil(len(d_l1_lte) / tpb[0])),)
    bound_helper[bpg, tpb](d_l1_lte,
                           d_l1_gte,
                           d_l1_lb,
                           d_l1_ub,
                           d_lbs,
                           d_ubs)
    return d_lbs, d_ubs

def relu_propagate_l1_GPU(l1_lte, l1_gte):
    @cuda.jit
    def relu_prop_helper(l1_lte, l1_gte, lbs, ubs, l1_relu_lte, l1_relu_gte):
        id = cuda.grid(1)
        if (id >= len(ubs)):
            return
        if (ubs[id] < 0):
            return  # as initialized with zeros
        if (lbs[id] > 0):
            for i in range(len(l1_lte[id])):
                l1_relu_lte[id][i] = l1_lte[id][i]
                l1_relu_gte[id][i] = l1_gte[id][i]
            return
        slope = ubs[id] / (ubs[id] - lbs[id])
        y_coeff = -ubs[id] * lbs[id] / (ubs[id] - lbs[id])
        for j in range(len(l1_lte[id])):
            l1_relu_gte[id][j] = slope * l1_gte[id][j]
        l1_relu_gte[id][0] += y_coeff

    d_l1_lte = cp.asarray(l1_lte)
    d_l1_gte = cp.asarray(l1_gte)
    d_lbs, d_ubs = get_bounds_GPU(d_l1_lte, d_l1_gte)
    d_l1_relu_lte = cp.zeros(l1_lte.shape)
    d_l1_relu_gte = cp.zeros(l1_lte.shape)
    tpb = (min(1024, len(l1_lte)),)
    bpg = (int(np.ceil(len(l1_lte) / tpb[0])),)
    relu_prop_helper[bpg, tpb](d_l1_lte,
                               d_l1_gte,
                               d_lbs,
                               d_ubs,
                               d_l1_relu_lte,
                               d_l1_relu_gte)
    l1_relu_gte = cp.asnumpy(d_l1_relu_gte)
    l1_relu_lte = cp.asnumpy(d_l1_relu_lte)
    return l1_relu_lte, l1_relu_gte

def relu_propagate_l1_GPU2(l1_lte, l1_gte):
    @cuda.jit
    def relu_prop_helper2(l1_lte, l1_gte, lbs, ubs, l1_relu_lte, l1_relu_gte):
        idx, idy = cuda.grid(2)
        if (idx >= len(ubs) or idy >= len(l1_lte[idx])):
            return
        if (ubs[idx] < 0):
            return  # as initialized with zeros
        if (lbs[idx] > 0):
            l1_relu_lte[idx][idy] = l1_lte[idx][idy]
            l1_relu_gte[idx][idy] = l1_gte[idx][idy]
            return
        slope = ubs[idx] / (ubs[idx] - lbs[idx])
        l1_relu_gte[idx][idy] = slope * l1_gte[idx][idy]
        if (idy == 0):
            y_coeff = -ubs[idx] * lbs[idx] / (ubs[idx] - lbs[idx])
            l1_relu_gte[idx][0] += y_coeff

    d_l1_lte = cp.asarray(l1_lte)
    d_l1_gte = cp.asarray(l1_gte)
    d_lbs, d_ubs = get_bounds_GPU(d_l1_lte, d_l1_gte)
    d_l1_relu_lte = cp.zeros(l1_lte.shape)
    d_l1_relu_gte = cp.zeros(l1_lte.shape)
    tpb = (min(32, len(l1_lte)), min(32, len(l1_lte[0])))
    bpg = (int(np.ceil(len(l1_lte) / tpb[0])), int(np.ceil(len(l1_lte[0]) / tpb[0])))
    relu_prop_helper2[bpg, tpb](d_l1_lte,
                                d_l1_gte,
                                d_lbs,
                                d_ubs,
                                d_l1_relu_lte,
                                d_l1_relu_gte)
    l1_relu_gte = cp.asnumpy(d_l1_relu_gte)
    l1_relu_lte = cp.asnumpy(d_l1_relu_lte)
    return l1_relu_lte, l1_relu_gte

def network_condense_GPU( nodes):
    # equation[n1][n2] stores the bias and coeff of nodes of previous layer to form x[n1][n2] in order
    # if_activation[n1][n2] stores if there is an activation on x[n1][n2] (only relu considered for now)
    NO_OF_LAYERS = 3
    MAX_NODES_IN_LAYER = 2
    ineq_lte = np.zeros((NO_OF_LAYERS + 1, MAX_NODES_IN_LAYER + 1, MAX_NODES_IN_LAYER + 1))
    ineq_gte = np.zeros((NO_OF_LAYERS + 1, MAX_NODES_IN_LAYER + 1, MAX_NODES_IN_LAYER + 1))
    ineq_relu_gte = np.zeros((NO_OF_LAYERS + 1, MAX_NODES_IN_LAYER + 1, MAX_NODES_IN_LAYER + 1))
    ineq_relu_lte = np.zeros((NO_OF_LAYERS + 1, MAX_NODES_IN_LAYER + 1, MAX_NODES_IN_LAYER + 1))
    if_activation = np.zeros((NO_OF_LAYERS + 1, MAX_NODES_IN_LAYER + 1))
    # TO-DO: can make the assumption if one node in a layer has relu then all do
    for current in nodes:
        if isinstance(current, Function):
            for (node, eq) in zip(current.stmts[0], current.stmts[1]):
                coeffs = np.zeros((MAX_NODES_IN_LAYER + 1,))
                eq = texpr_to_dict(eq)
                for var, val in eq.items():
                    if var != '_':
                        var = int(
                            var[2])  # TO-DO: do something more general than assume format X[N][N] for var name
                        coeffs[var] = val
                    else:
                        coeffs[0] = val
                ineq_lte[int(str(node)[1]), int(str(node)[2]),
                :] = coeffs  # TO-DO: do something more general than assume format X[N][N] for var name
        elif isinstance(current, Activation):
            if_activation[int(str(current.stmts)[1]), int(str(current.stmts)[2])] = True
        else:
            '''     What to do here
            for stmt in reversed(current.stmts):
                state = semantics.assume_call_semantics(stmt, state, manager)'''
            continue

    # can remove layer 0 as it has no inequations
    for i in range(1, len(ineq_lte)):

        print(f"\t\t LAYER {i} Input Equations")
        for j in range(1, len(ineq_lte[0])):
            print(
                f"Node: {j} -> if_activation: {if_activation[i, j]}\n eq: {ineq_str(ineq_lte[i, j], i, j, '=', i - 1)} ")

        if (i != 1):
            ineq_lte[i], ineq_gte[i] = back_propagate_l1_GPU(ineq_lte[i], ineq_relu_lte[i - 1],
                                                                  ineq_relu_gte[i - 1])
        else:
            ineq_gte[i] = ineq_lte[i].copy()

        print(f"\t\t LAYER {i} Substituted")
        for j in range(1, len(ineq_lte[0])):
            print(f"\tNode {j}")
            print(f" eq LTE L1: {ineq_str(ineq_lte[i][j], i, j, '>=', 0)}")
            print(f" eq GTE L1: {ineq_str(ineq_gte[i][j], i, j, '<=', 0)}")
            print(
                f" eq (LB,UB): {get_bounds_single(ineq_lte[i], ineq_gte[i], j)}")  # Performing the whole debug-print segment in CPU will be removed later.

        if (if_activation[i,1]==1):  # assuming if first node in a layer has activation then all do
            ineq_relu_lte[i], ineq_relu_gte[i] = relu_propagate_l1_GPU(ineq_lte[i], ineq_gte[i])
            print(f"\t RELU-LAYER {i}")
            for j in range(1, len(ineq_lte[0])):
                print(f"\tNode {j}")
                print(f" Relu eq LTE L1: {ineq_str(ineq_relu_lte[i][j], i, j, '>=', 0)}")
                print(f" Relu eq GTE L1: {ineq_str(ineq_relu_gte[i][j], i, j, '<=', 0)}")
                print(f"Relu eq (LB,UB): {get_bounds_single(ineq_relu_lte[i], ineq_relu_gte[i], j)}")
            # print stuff
        else:
            ineq_relu_lte[i][j], ineq_relu_gte[i][j] = ineq_lte[i][j], ineq_gte[i][j]
            print(f"\t\t NO RELU ON LAYER {i}")
