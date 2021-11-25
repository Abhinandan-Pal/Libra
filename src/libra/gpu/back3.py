import time
import numpy as np
import cupy as cp
from numba import cuda
def back_propagate_l1_CPU(eq_layer, ineq_prev_lte,
                       ineq_prev_gte, node_num):
    ln_coeff = eq_layer[node_num]  # store the bias term
    l1_lte = [0] * len(ineq_prev_lte[1])
    l1_gte = [0] * len(ineq_prev_lte[1])
    l1_lte[0] = ln_coeff[0]
    l1_gte[0] = ln_coeff[0]
    for i in range(1, len(l1_lte)):
        for p in range(0, len(ineq_prev_lte[1])):
            if(ln_coeff[i]>0):
                l1_lte[p] += ln_coeff[i] * ineq_prev_lte[i-1][p]            #should it be i or i-1?
                l1_gte[p] += ln_coeff[i] * ineq_prev_gte[i-1][p]
            else:
                l1_lte[p] += ln_coeff[i] * ineq_prev_gte[i-1][p]
                l1_gte[p] += ln_coeff[i] * ineq_prev_lte[i-1][p]
    return l1_lte, l1_gte
@cuda.jit
def helper(l1_lte,l1_gte,coeff,ineq_prev_lte_i,ineq_prev_gte_i):
    id = cuda.grid(1)
    if(coeff[0]>0):
        l1_lte[id] += coeff[0] * ineq_prev_lte_i[id]            #should it be i or i-1?
        l1_gte[id] += coeff[0] * ineq_prev_gte_i[id]
    else:
        l1_lte[id] += coeff[0] * ineq_prev_gte_i[id]
        l1_gte[id] += coeff[0] * ineq_prev_lte_i[id]

def back_propagate_l1_GPU(eq_layer, ineq_prev_lte,
                       ineq_prev_gte, node_num):
    
    ln_coeff = eq_layer[node_num]  # store the bias term
    d_l1_lte = cp.zeros(len(ineq_prev_lte[1])) 
    d_l1_gte = cp.zeros(len(ineq_prev_lte[1]))
    d_l1_lte[0] = cp.asarray(ln_coeff[0])
    d_l1_gte[0] = cp.asarray(ln_coeff[0])
    cuda_iters = len(ineq_prev_lte[1])
    for i in range(1, len(d_l1_lte)):
        d_coeff = cp.array([ln_coeff[i]]).astype(np.float32)

        d_ineq_prev_lte_i = cp.asarray(ineq_prev_lte[i-1])      #only load whats needed now to prevent out of memory

        d_ineq_prev_gte_i = cp.asarray(ineq_prev_gte[i-1])
        
        tpb = (min(16,cuda_iters),)
        bpg = (int(np.ceil(cuda_iters/tpb[0])),)
        helper[bpg,tpb](d_l1_lte,
            d_l1_gte,
            d_coeff,
            d_ineq_prev_lte_i,
            d_ineq_prev_gte_i,)
        #pycuda.driver.Context.synchronize();
    l1_gte = cp.asnumpy(d_l1_gte)
    l1_lte = cp.asnumpy(d_l1_lte)
    return l1_lte, l1_gte

def caller():
    MAX_NODES_IN_LAYER = 5

    curr_lte = np.random.randn(MAX_NODES_IN_LAYER, MAX_NODES_IN_LAYER+1).astype(np.float32)
    curr_gte  = np.random.randn(MAX_NODES_IN_LAYER, MAX_NODES_IN_LAYER+1).astype(np.float32)
    
    prev_lte = np.random.randn(MAX_NODES_IN_LAYER, MAX_NODES_IN_LAYER+1).astype(np.float32)
    prev_gte  = np.random.randn(MAX_NODES_IN_LAYER, MAX_NODES_IN_LAYER+1).astype(np.float32)
    print(f"MAX_NODES_IN_LAYER = {MAX_NODES_IN_LAYER}")
    time_sec = time.time()
    for i in range(len(curr_lte)):
        print(f"NODE{i}\n{back_propagate_l1_CPU(curr_lte, prev_lte, prev_gte , i)}");
    time_sec = time.time() - time_sec
    print(f"CPU time: {time_sec}\n\n")
    time_sec = time.time()
    for i in range(len(curr_lte)):
        print(f"Node{i}\n{back_propagate_l1_GPU(curr_lte, prev_lte, prev_gte , i)}");

    time_sec = time.time() - time_sec
    print(f"GPU time: {time_sec}")

caller()