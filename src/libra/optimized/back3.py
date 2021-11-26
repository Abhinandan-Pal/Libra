import time
import numpy as np
import cupy as cp
from numba import cuda


def back_propagate_l1_CPU(ln_coeff, ineq_prev_lte,
                          ineq_prev_gte):
    l1_lte = np.zeros((len(ln_coeff), len(ineq_prev_lte[1])))
    l1_gte = np.zeros((len(ln_coeff), len(ineq_prev_lte[1])))
    l1_lte[:, 0] = ln_coeff[:, 0]
    l1_gte[:, 0] = ln_coeff[:, 0]
    for i in range(1, len(l1_lte)):
        for k in range(0, len(ln_coeff)):
            for p in range(0, len(ineq_prev_lte[1])):
                if (ln_coeff[k][i] > 0):
                    l1_lte[k][p] += ln_coeff[k][i] * ineq_prev_lte[i][p]  # should it be i or i-1?
                    l1_gte[k][p] += ln_coeff[k][i] * ineq_prev_gte[i][p]

                else:
                    l1_lte[k][p] += ln_coeff[k][i] * ineq_prev_gte[i][p]
                    l1_gte[k][p] += ln_coeff[k][i] * ineq_prev_lte[i][p]
    return l1_lte, l1_gte


@cuda.jit
def helper(l1_lte, l1_gte, coeff, ineq_prev_lte_i, ineq_prev_gte_i):
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


def back_propagate_l1_GPU(ln_coeff, ineq_prev_lte,
                          ineq_prev_gte):
    d_l1_lte = cp.zeros((len(ln_coeff), len(ineq_prev_lte[1])))
    d_l1_gte = cp.zeros((len(ln_coeff), len(ineq_prev_lte[1])))
    d_l1_lte[:, 0] = cp.asarray(ln_coeff[:, 0])
    d_l1_gte[:, 0] = cp.asarray(ln_coeff[:, 0])
    cuda_iters = (len(ln_coeff), len(ineq_prev_lte[1]))
    tpb = (min(32, cuda_iters[0]), min(32, cuda_iters[1]))
    bpg = (int(np.ceil(cuda_iters[0] / tpb[0])), int(np.ceil(cuda_iters[1] / tpb[1])))
    print(f"tpb = {tpb}, bpg = {bpg}")
    for i in range(1, len(d_l1_lte)):
        d_coeff = cp.array(ln_coeff[:, i]).astype(np.float32)

        d_ineq_prev_lte_i = cp.asarray(ineq_prev_lte[i])  # only load whats needed now to prevent out of memory

        d_ineq_prev_gte_i = cp.asarray(ineq_prev_gte[i])

        helper[bpg, tpb](d_l1_lte,
                         d_l1_gte,
                         d_coeff,
                         d_ineq_prev_lte_i,
                         d_ineq_prev_gte_i, )
        # pycuda.driver.Context.synchronize();
    l1_gte = cp.asnumpy(d_l1_gte)
    l1_lte = cp.asnumpy(d_l1_lte)
    return l1_lte, l1_gte


def caller():
    MAX_NODES_IN_LAYER = 100

    curr_lte = np.random.randn(MAX_NODES_IN_LAYER, MAX_NODES_IN_LAYER + 1).astype(np.float32)
    curr_gte = np.random.randn(MAX_NODES_IN_LAYER, MAX_NODES_IN_LAYER + 1).astype(np.float32)

    prev_lte = np.random.randn(MAX_NODES_IN_LAYER, MAX_NODES_IN_LAYER + 1).astype(np.float32)
    prev_gte = np.random.randn(MAX_NODES_IN_LAYER, MAX_NODES_IN_LAYER + 1).astype(np.float32)
    print(f"MAX_NODES_IN_LAYER = {MAX_NODES_IN_LAYER}")
    time_sec = time.time()
    print(f"{back_propagate_l1_CPU(curr_lte, prev_lte, prev_gte)[1][1][1]}")
    time_sec = time.time() - time_sec
    print(f"CPU time: {time_sec}\n\n")

    time_sec = time.time()
    print(f"{back_propagate_l1_GPU(curr_lte, prev_lte, prev_gte)[1][1][1]}")

    time_sec = time.time() - time_sec
    print(f"GPU time: {time_sec}")


caller()