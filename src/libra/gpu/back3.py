import time
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

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

def back_propagate_l1_GPU(eq_layer, ineq_prev_lte,
                       ineq_prev_gte, node_num):
    
    cuda_template = """
    __global__ void gpu_helper(float *l1_lte,float *l1_gte,float coeff,float *ineq_prev_lte_i,float *ineq_prev_gte_i)
    {
        int id = blockIdx.x*blockDim.x + threadIdx.x;
        if(coeff>0)
        {
            l1_lte[id] += coeff * ineq_prev_lte_i[id];
            l1_gte[id] += coeff * ineq_prev_gte_i[id];
        }
        else
        {
            l1_lte[id] += coeff * ineq_prev_gte_i[id];
            l1_gte[id] += coeff * ineq_prev_lte_i[id];
        }
        
    }
    """
    cuda_code = cuda_template % {}
    mod = SourceModule(cuda_code)
    gpu_helper = mod.get_function("gpu_helper")

    ln_coeff = eq_layer[node_num]  # store the bias term
    l1_lte = np.zeros(len(ineq_prev_lte[1])) 
    l1_gte = np.zeros(len(ineq_prev_lte[1]))
    l1_lte[0] = ln_coeff[0]
    l1_gte[0] = ln_coeff[0]

    cuda_iters = len(ineq_prev_lte[1])

    d_l1_lte = cuda.mem_alloc(l1_lte.nbytes)
    cuda.memcpy_htod(d_l1_lte, l1_lte)

    d_l1_gte = cuda.mem_alloc(l1_gte.nbytes)
    cuda.memcpy_htod(d_l1_gte, l1_gte)

    

    d_ln_coeff = cuda.mem_alloc(ln_coeff.nbytes)
    cuda.memcpy_htod(d_ln_coeff, ln_coeff)

    for i in range(1, len(l1_lte)):
        d_coeff = cuda.mem_alloc(ln_coeff[i].nbytes)
        cuda.memcpy_htod(d_coeff, ln_coeff[i])

        d_ineq_prev_lte_i = cuda.mem_alloc(ineq_prev_lte[i-1].nbytes)
        cuda.memcpy_htod(d_ineq_prev_lte_i, ineq_prev_lte[i-1])

        d_ineq_prev_gte_i = cuda.mem_alloc(ineq_prev_gte[i-1].nbytes)
        cuda.memcpy_htod(d_ineq_prev_gte_i, ineq_prev_gte[i-1])
        
        gpu_helper(d_l1_lte,
            d_l1_gte,
            d_coeff,
            d_ineq_prev_lte_i,
            d_ineq_prev_gte_i,
            block = (min(1024,cuda_iters),1, 1),
            grid = (int(cuda_iters + 1023 / 1024),1,1))
        pycuda.driver.Context.synchronize();
    return l1_lte, l1_gte

def caller():
    MAX_NODES_IN_LAYER = 200

    curr_lte = np.random.randn(MAX_NODES_IN_LAYER, MAX_NODES_IN_LAYER+1).astype(np.float32)
    curr_gte  = np.random.randn(MAX_NODES_IN_LAYER, MAX_NODES_IN_LAYER+1).astype(np.float32)
    
    prev_lte = np.random.randn(MAX_NODES_IN_LAYER, MAX_NODES_IN_LAYER+1).astype(np.float32)
    prev_gte  = np.random.randn(MAX_NODES_IN_LAYER, MAX_NODES_IN_LAYER+1).astype(np.float32)
    print(f"MAX_NODES_IN_LAYER = {MAX_NODES_IN_LAYER}")
    '''time_sec = time.time()
    for i in range(len(curr_lte)):
        a,b = back_propagate_l1_CPU(curr_lte, prev_lte, prev_gte , i)

    time_sec = time.time() - time_sec
    print(f"CPU time: {time_sec}")'''

    time_sec = time.time()
    for i in range(len(curr_lte)):
        a,b = back_propagate_l1_GPU(curr_lte, prev_lte, prev_gte , i)

    time_sec = time.time() - time_sec
    print(f"GPU time: {time_sec}")

caller()