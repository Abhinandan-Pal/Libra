import time
import numpy as np
import cupy as cp
from numba import cuda
def get_bounds_CPU(l1_layer_lte,l1_layer_gte,node_num , l1_lb = -1,l1_ub = 1):
    l1_lte = l1_layer_lte[node_num]
    l1_gte = l1_layer_gte[node_num]
    lb = l1_lte[0]
    for coeff in l1_lte[1:]:
        if (coeff < 0):
            lb += coeff * l1_ub
        else:
            lb += coeff* l1_lb
    ub = l1_gte[0]
    for coeff in l1_gte[1:]:
        if (coeff > 0):
            ub += coeff * l1_ub
        else:
            ub += coeff* l1_lb
    return lb, ub

def relu_propagate_l1_CPU(l1_layer_lte,l1_layer_gte,node_num):
    l1_lte = l1_layer_lte[node_num]
    l1_gte = l1_layer_gte[node_num]
    lb,ub = get_bounds_CPU(l1_layer_lte,l1_layer_gte,node_num)
    l1_relu_lte = [0] * len(l1_lte)
    l1_relu_gte = [0] * len(l1_gte)
    '''Case 1(Strictly Negative)'''
    if(ub<0):
        return l1_relu_lte,l1_relu_gte
    '''Case 2(Strictly Positive)'''
    if(lb > 0):
        return l1_lte, l1_gte
    '''Case 3(Crossing Relu)'''
    slope = ub/(ub-lb)
    y_coeff = -ub*lb/(ub-lb)
    for i in range(len(l1_gte)):
        l1_relu_gte[i] = slope*l1_gte[i]
    l1_relu_gte[0]+= y_coeff
    return l1_relu_lte,l1_relu_gte

def get_bounds_CPU2(l1_lte,l1_gte , l1_lb = -1,l1_ub = 1):
    lbs = np.zeros(l1_lte.shape[0])
    ubs = np.zeros(l1_lte.shape[0])
    for i in range(len(l1_lte)):
        lbs[i] = l1_lte[i][0]
        for coeff in l1_lte[i][1:]:
            if (coeff < 0):
                lbs[i] += coeff * l1_ub
            else:
                lbs[i] += coeff* l1_lb
        ubs[i] = l1_gte[i][0]
        for coeff in l1_gte[i][1:]:
            if (coeff > 0):
                ubs[i] += coeff * l1_ub
            else:
                ubs[i] += coeff* l1_lb
    return lbs, ubs

def relu_propagate_l1_CPU2(l1_lte,l1_gte):
    lbs,ubs = get_bounds_CPU2(l1_lte,l1_gte)
    l1_relu_lte = np.zeros(l1_lte.shape)
    l1_relu_gte = np.zeros(l1_lte.shape)
    for i in range(len(l1_lte)):
        
        if(ubs[i] < 0):
            continue
        elif(lbs[i] > 0):  
            l1_relu_lte[i] = l1_lte[i]
            l1_relu_gte[i] = l1_gte[i]
        else:
            slope = ubs[i]/(ubs[i]-lbs[i])
            y_coeff = -ubs[i]*lbs[i]/(ubs[i]-lbs[i])
            for j in range(len(l1_lte[i])):
                l1_relu_gte[i][j] = slope*l1_gte[i][j]
            l1_relu_gte[i][0]+= y_coeff
    return l1_relu_lte,l1_relu_gte

@cuda.jit
def bound_helper(l1_lte,l1_gte ,l1_lb,l1_ub,lbs,ubs):
    id = cuda.grid(1)
    if(id>= len(lbs)):
      return
    lbs[id] = l1_lte[id][0]
    for coeff in l1_lte[id,1:]:
        if (coeff < 0):
            lbs[id] += coeff * l1_ub[0]
        else:
            lbs[id] += coeff* l1_lb[0]
    ubs[id] = l1_gte[id][0]
    for coeff in l1_gte[id,1:]:
        if (coeff > 0):
            ubs[id] += coeff * l1_ub[0]
        else:
            ubs[id] += coeff* l1_lb[0]

def get_bounds_GPU(d_l1_lte,d_l1_gte , l1_lb = -1,l1_ub = 1):
    d_l1_lb = cp.asarray([l1_lb])
    d_l1_ub = cp.asarray([l1_ub])
    d_lbs = cp.zeros(d_l1_lte.shape[0])
    d_ubs = cp.zeros(d_l1_lte.shape[0])
    tpb = (min(1024,len(d_l1_lte)),)
    bpg = (int(np.ceil(len(d_l1_lte)/tpb[0])),)
    bound_helper[bpg,tpb](d_l1_lte,
        d_l1_gte ,
        d_l1_lb,
        d_l1_ub,
        d_lbs,
        d_ubs)
    return d_lbs, d_ubs

@cuda.jit
def relu_prop_helper(l1_lte,l1_gte ,lbs,ubs,l1_relu_lte,l1_relu_gte):
    id = cuda.grid(1)
    if(id>= len(ubs)):
      return
    if(ubs[id] < 0):
          return  # as initialized with zeros
    if(lbs[id] > 0):
        for i in range(len(l1_lte[id])):
          l1_relu_lte[id][i] = l1_lte[id][i]
          l1_relu_gte[id][i] = l1_gte[id][i]
        return
    slope = ubs[id]/(ubs[id]-lbs[id])
    y_coeff = -ubs[id]*lbs[id]/(ubs[id]-lbs[id])
    for j in range(len(l1_lte[id])):
        l1_relu_gte[id][j] = slope*l1_gte[id][j]
    l1_relu_gte[id][0]+= y_coeff


def relu_propagate_l1_GPU(l1_lte,l1_gte):
    d_l1_lte = cp.asarray(l1_lte)
    d_l1_gte = cp.asarray(l1_gte)
    d_lbs,d_ubs = get_bounds_GPU(d_l1_lte,d_l1_gte)
    d_l1_relu_lte = cp.zeros(l1_lte.shape)
    d_l1_relu_gte = cp.zeros(l1_lte.shape)
    tpb = (min(1024,len(l1_lte)),)
    bpg = (int(np.ceil(len(l1_lte)/tpb[0])),)
    relu_prop_helper[bpg,tpb](d_l1_lte,
        d_l1_gte ,
        d_lbs,
        d_ubs,
        d_l1_relu_lte,
        d_l1_relu_gte)
    l1_relu_gte = cp.asnumpy(d_l1_relu_gte)
    l1_relu_lte = cp.asnumpy(d_l1_relu_lte)
    return l1_relu_lte,l1_relu_gte
@cuda.jit
def relu_prop_helper2(l1_lte,l1_gte ,lbs,ubs,l1_relu_lte,l1_relu_gte):
    idx,idy = cuda.grid(2)
    if(idx>= len(ubs) or idy>=len(l1_lte[idx])):
      return
    if(ubs[idx] < 0):
          return  # as initialized with zeros
    if(lbs[idx] > 0):
        l1_relu_lte[idx][idy] = l1_lte[idx][idy]
        l1_relu_gte[idx][idy] = l1_gte[idx][idy]
        return
    slope = ubs[idx]/(ubs[idx]-lbs[idx])
    l1_relu_gte[idx][idy] = slope*l1_gte[idx][idy]
    if(idy == 0):
        y_coeff = -ubs[idx]*lbs[idx]/(ubs[idx]-lbs[idx])
        l1_relu_gte[idx][0]+= y_coeff


def relu_propagate_l1_GPU2(l1_lte,l1_gte):
    d_l1_lte = cp.asarray(l1_lte)
    d_l1_gte = cp.asarray(l1_gte)
    d_lbs,d_ubs = get_bounds_GPU(d_l1_lte,d_l1_gte)
    d_l1_relu_lte = cp.zeros(l1_lte.shape)
    d_l1_relu_gte = cp.zeros(l1_lte.shape)
    tpb = (min(32,len(l1_lte)),min(32,len(l1_lte[0])))
    bpg = (int(np.ceil(len(l1_lte)/tpb[0])),int(np.ceil(len(l1_lte[0])/tpb[0])))
    relu_prop_helper2[bpg,tpb](d_l1_lte,
        d_l1_gte ,
        d_lbs,
        d_ubs,
        d_l1_relu_lte,
        d_l1_relu_gte)
    l1_relu_gte = cp.asnumpy(d_l1_relu_gte)
    l1_relu_lte = cp.asnumpy(d_l1_relu_lte)
    return l1_relu_lte,l1_relu_gte

def caller2():
    MAX_NODES_IN_LAYER =1000

    curr_lte = np.random.randn(MAX_NODES_IN_LAYER, MAX_NODES_IN_LAYER+1).astype(np.float32)
    curr_gte  = np.random.randn(MAX_NODES_IN_LAYER, MAX_NODES_IN_LAYER+1).astype(np.float32)

    print(f"MAX_NODES_IN_LAYER = {MAX_NODES_IN_LAYER}")
    '''time_sec = time.time()
    for i in range(len(curr_lte)):
        print(f"NODE{i}\n{relu_propagate_l1_CPU(curr_lte, curr_gte , i)}");
    time_sec = time.time() - time_sec
    print(f"CPU time: {time_sec}\n\n")'''

    '''time_sec = time.time()
    print(f"{relu_propagate_l1_CPU2(curr_lte, curr_gte )[1][1][1]}")
    time_sec = time.time() - time_sec
    print(f"CPU2 time: {time_sec}\n\n")'''
    time_sec = time.time()
    print(f"{relu_propagate_l1_GPU(curr_lte, curr_gte )[1][1][1]}")

    time_sec = time.time() - time_sec
    print(f"GPU1 time: {time_sec}")
    
    time_sec = time.time()
    print(f"{relu_propagate_l1_GPU2(curr_lte, curr_gte )[1][1][1]}")

    time_sec = time.time() - time_sec
    print(f"GPU1 time: {time_sec}")

caller2()
