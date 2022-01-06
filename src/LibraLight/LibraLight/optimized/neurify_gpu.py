import numpy as np
import cupy as cp
from numba import cuda
from optimized import commons
import warnings
from colorama import Style, Fore

def get_bounds_GPU(d_l1_lte, d_l1_gte, d_l1_lb, d_l1_ub):
    @cuda.jit
    def bound_helper(l1_lte, l1_gte, l1_lb, l1_ub, lbs, ubs):
        init_id,id = cuda.grid(2)
        if (id >= len(lbs[0]) or init_id >= len(lbs)):
            return
        lbs[init_id][id] = l1_lte[init_id,id, 0]
        for i in range(1,len(l1_lte[init_id,id])):
            if(l1_lte[init_id][id,i]<0):
                lbs[init_id][id] += l1_lte[init_id,id,i] * l1_ub[init_id][i]
            else:
                lbs[init_id][id] += l1_lte[init_id,id,i] * l1_lb[init_id][i]
        ubs[init_id][id] = l1_gte[init_id,id, 0]
        for i in range(1,len(l1_gte[init_id,id])):
            if(l1_gte[init_id,id,i]>0):
                ubs[init_id][id] += l1_gte[init_id,id,i] * l1_ub[init_id][i]
            else:
                ubs[init_id][id] += l1_gte[init_id,id,i] * l1_lb[init_id][i]

    d_lbs = cp.zeros(d_l1_lte.shape[0:2])
    d_ubs = cp.zeros(d_l1_lte.shape[0:2])
    cuda_iters1 = (len(d_l1_lte), len(d_l1_lte[0]))
    tpb1 = (min(64, cuda_iters1[0]), min(16, cuda_iters1[1]))
    bpg1 = (int(np.ceil(cuda_iters1[0] / tpb1[0])), int(np.ceil(cuda_iters1[1] / tpb1[1])))
    bound_helper[bpg1, tpb1](d_l1_lte,
                             d_l1_gte,
                             d_l1_lb,
                             d_l1_ub,
                             d_lbs,
                             d_ubs)
    return d_lbs, d_ubs

def get_bounds_single_neurify(l1_layer_lte, l1_layer_gte, node_num: int, l1_lb, l1_ub, relu_val=[1, 0, 1, 0]):
    l1_lte = l1_layer_lte[node_num] * relu_val[0]
    l1_lte[0] += relu_val[1]
    l1_gte = l1_layer_gte[node_num] * relu_val[2]
    l1_gte[0] += relu_val[3]
    lb = l1_lte[0]
    for i in range(1, len(l1_lb)):
        if (l1_lte[i] < 0):
            lb += l1_lte[i] * l1_ub[i][0]
        else:
            lb += l1_lte[i] * l1_lb[i][0]
    ub = l1_gte[0]
    for i in range(1, len(l1_ub)):
        if (l1_gte[i] > 0):
            ub += l1_gte[i] * l1_ub[i][1]
        else:
            ub += l1_gte[i] * l1_lb[i][1]
    return lb, ub

def relu_compute_GPU(d_lbs_low, d_ubs_low,d_lbs_up, d_ubs_up, d_relu_layer, d_active_pattern,d_l1_lb,d_l1_ub,d_if_activation):
    @cuda.jit
    def relu_compute_helper(lbs_up, ubs_up,lbs_low,ubs_low, relu_layer, active_pattern,if_activation):
        init_id,id = cuda.grid(2)
        if (id< 1 or id >= len(ubs_up[0]) or init_id >= len(ubs_up)):
            return
        if(if_activation[id] == 0):
            active_pattern[init_id][id] = 2
            relu_layer[init_id][id] = (1.0, 0.0, 1.0, 0.0)
            return
        if(ubs_up[init_id][id] < 0):
            active_pattern[init_id][id] = 0
        elif(lbs_low[init_id][id] >= 0):
            active_pattern[init_id][id] = 1
        else:
            active_pattern[init_id][id] = 2
        relu_layer[init_id][id][1] = 0.0
        if (ubs_low[init_id][id] <= 0):
            relu_layer[init_id][id][0] = 0.0
        elif (lbs_low[init_id][id] >= 0):
            relu_layer[init_id][id][0] = 1.0
        else:
            slope = ubs_low[init_id][id] / (ubs_low[init_id][id] - lbs_low[init_id][id])
            relu_layer[init_id][id][0] = slope
        relu_layer[init_id][id][3] = 0.0
        if (ubs_up[init_id][id] <= 0):
            relu_layer[init_id][id][2] = 0.0
        elif (lbs_up[init_id][id] >= 0):
            relu_layer[init_id][id][2] = 1.0
        else:
            slope = ubs_up[init_id][id] / (ubs_up[init_id][id] - lbs_up[init_id][id])
            y_coeff = -ubs_up[init_id][id] * lbs_up[init_id][id] / (ubs_up[init_id][id] - lbs_up[init_id][id])
            relu_layer[init_id][id][2] = slope
            relu_layer[init_id][id][3] = y_coeff

    cuda_iters1 = (len(d_lbs_low), len(d_lbs_low[0]))
    tpb1 = (min(64, cuda_iters1[0]), min(16, cuda_iters1[1]))
    bpg1 = (int(np.ceil(cuda_iters1[0] / tpb1[0])), int(np.ceil(cuda_iters1[1] / tpb1[1])))
    relu_compute_helper[bpg1, tpb1](d_lbs_up,
                                  d_ubs_up,
                                  d_lbs_low,
                                  d_ubs_low,
                                  d_relu_layer,
                                  d_active_pattern,
                                  d_if_activation)
    return d_relu_layer, d_active_pattern

@cuda.jit
def back_affine_helper(i, l1_lte, l1_gte, ln_coeff_lte, ln_coeff_gte, ineq_prev_lte, ineq_prev_gte):
    init_id,k, p = cuda.grid(3)
    i = i[0]
    if (k >= len(l1_lte[0]) or p >= len(ineq_prev_lte[i]) or init_id >= len(l1_lte)):
        return
    if (ln_coeff_lte[init_id][k][i] > 0):
        l1_lte[init_id][k][p] += ln_coeff_lte[init_id][k][i] * ineq_prev_lte[i][p]  # should it be i or i-1?
    else:
        l1_lte[init_id][k][p] += ln_coeff_lte[init_id][k][i] * ineq_prev_gte[i][p]
    if (ln_coeff_gte[init_id][k][i] > 0):
        l1_gte[init_id][k][p] += ln_coeff_gte[init_id][k][i] * ineq_prev_gte[i][p]
    else:
        l1_gte[init_id][k][p] += ln_coeff_gte[init_id][k][i] * ineq_prev_lte[i][p]

def back_affine_GPU(d_ineq_prev_lte, d_ineq_prev_gte, d_ln_coeff_lte, d_ln_coeff_gte):
    ln_shape = (len(d_ln_coeff_lte),) + d_ineq_prev_gte.shape
    l1_lte = np.zeros(ln_shape)
    l1_gte = np.zeros(ln_shape)

    l1_lte[:, :, 0] = d_ln_coeff_lte.get()[:, :, 0]
    l1_gte[:, :, 0] = d_ln_coeff_gte.get()[:, :, 0]
    d_l1_lte = cp.asarray(l1_lte)
    d_l1_gte = cp.asarray(l1_gte)

    cuda_iters = (len(d_l1_lte), len(d_l1_lte[0]), len(d_ineq_prev_lte[0]))
    tpb = (min(16, cuda_iters[0]), min(8, cuda_iters[1]), min(8, cuda_iters[2]))
    bpg = (int(np.ceil(cuda_iters[0] / tpb[0])), int(np.ceil(cuda_iters[1] / tpb[1])), int(np.ceil(cuda_iters[2] / tpb[2])))
    for i in range(1, len(d_l1_lte[0])):
        d_i = cp.array([i])
        back_affine_helper[bpg, tpb](d_i, d_l1_lte,
                                     d_l1_gte,
                                     d_ln_coeff_lte,
                                     d_ln_coeff_gte,
                                     d_ineq_prev_lte,
                                     d_ineq_prev_gte)
    return d_l1_lte, d_l1_gte

@cuda.jit
def back_relu_coeff_helper(relu_layer, ln_coeff_lte, ln_coeff_gte):
    init_id,i, j = cuda.grid(3)
    if (i < 1 or j < 1 or i >= len(ln_coeff_lte[0]) or j >= len(relu_layer[0]) or init_id >= len(ln_coeff_lte)):
        return
    if (ln_coeff_lte[init_id][i][j] > 0):
        ln_coeff_lte[init_id][i][j] = relu_layer[init_id][j][0] * ln_coeff_lte[init_id][i][j]
    else:
        ln_coeff_lte[init_id][i][j] = relu_layer[init_id][j][2] * ln_coeff_lte[init_id][i][j]
    if (ln_coeff_gte[init_id][i][j] > 0):
        ln_coeff_gte[init_id][i][j] = relu_layer[init_id][j][2] * ln_coeff_gte[init_id][i][j]
    else:
        ln_coeff_gte[init_id][i][j] = relu_layer[init_id][j][0] * ln_coeff_gte[init_id][i][j]

@cuda.jit
def back_relu_base_helper(relu_layer, ln_coeff_lte, ln_coeff_gte):
    init_id,i = cuda.grid(2)
    if (i < 1 or i >= len(ln_coeff_lte[0]) or init_id >= len(ln_coeff_lte)):
        return
    for j in range(1, len(relu_layer[0])):
        if (ln_coeff_lte[init_id][i][j] > 0):
            ln_coeff_lte[init_id][i][0] += relu_layer[init_id][j][1] * ln_coeff_lte[init_id][i][j]
        else:
            ln_coeff_lte[init_id][i][0] += relu_layer[init_id][j][3] * ln_coeff_lte[init_id][i][j]
        if (ln_coeff_gte[init_id][i][j] > 0):
            ln_coeff_gte[init_id][i][0] += relu_layer[init_id][j][3] * ln_coeff_gte[init_id][i][j]
        else:
            ln_coeff_gte[init_id][i][0] += relu_layer[init_id][j][1] * ln_coeff_gte[init_id][i][j]

def back_relu_GPU(d_relu_layer, d_ln_coeff_lte, d_ln_coeff_gte):
    cuda_iters1 = (len(d_ln_coeff_lte), len(d_ln_coeff_lte[0]))
    tpb1 = (min(64, cuda_iters1[0]), min(16, cuda_iters1[1]))
    bpg1 = (int(np.ceil(cuda_iters1[0] / tpb1[0])), int(np.ceil(cuda_iters1[1] / tpb1[1])))
    back_relu_base_helper[bpg1, tpb1](d_relu_layer,
                                      d_ln_coeff_lte,
                                      d_ln_coeff_gte)
    cuda_iters = (len(d_ln_coeff_lte), len(d_ln_coeff_lte[0]), len(d_relu_layer[0]))  # Untestd: d_relu_layer[0]
    tpb = (min(16, cuda_iters[1]), min(8, cuda_iters[1]), min(8, cuda_iters[2]))
    bpg = (
    int(np.ceil(cuda_iters[0] / tpb[0])), int(np.ceil(cuda_iters[1] / tpb[1])), int(np.ceil(cuda_iters[2] / tpb[2])))
    back_relu_coeff_helper[bpg, tpb](d_relu_layer,
                                     d_ln_coeff_lte,
                                     d_ln_coeff_gte)

def back_propagate_GPU1(d_ln_coeff_lte, d_ln_coeff_gte,d_affine, d_relu, layer: int, if_activation):
    while (layer != 1):  # layer zero is input and layer one is in already in terms of input
        # First relu of previous layer
        if (any(if_activation[layer - 1])):
            back_relu_GPU(d_relu[:,layer - 1], d_ln_coeff_lte, d_ln_coeff_gte)
        # Then affine of previous layer
        d_ineq_prev_gte = d_affine[layer - 1]
        d_ineq_prev_lte = d_affine[layer - 1]
        d_ln_coeff_lte, d_ln_coeff_gte = back_affine_GPU(d_ineq_prev_lte, d_ineq_prev_gte, d_ln_coeff_lte,
                                                         d_ln_coeff_gte)
        layer -= 1
    return d_ln_coeff_lte, d_ln_coeff_gte

def back_propagate_GPU(d_affine, d_relu, layer: int, if_activation):
    # shift the CP creation to caller.
    ln_shape = (len(d_relu),) + d_affine[layer].shape
    ln_coeff_lte = np.zeros(ln_shape)
    ln_coeff_lte[:] = d_affine[layer].get()
    d_ln_coeff_lte = cp.asarray(ln_coeff_lte)

    ln_coeff_gte = np.zeros(ln_shape)
    ln_coeff_gte[:] = d_affine[layer].get()
    d_ln_coeff_gte = cp.asarray(ln_coeff_gte)
    d_ln_coeff_lte, d_ln_coeff_gte = back_propagate_GPU1(d_ln_coeff_lte, d_ln_coeff_gte,d_affine, d_relu, layer, if_activation)

    return d_ln_coeff_lte, d_ln_coeff_gte

def oneOutput(d_affine,d_relu,if_activation,d_l1_lb,d_l1_ub,outNodes,inv_var_index,sensitive,print_mode):
    outcomes = [None] * len(d_relu)
    for out1 in outNodes:
        ln_shape = (len(d_relu),) + d_affine[0].shape
        ln_coeff_lte = np.zeros(ln_shape).astype('float32')
        for out2 in outNodes:
            if(out2 != out1):
                ln_coeff_lte[:,out2,out1] = 1
                ln_coeff_lte[:,out2,out2] = -1
        d_ln_coeff_lte = cp.asarray(ln_coeff_lte)
        d_ln_coeff_gte = d_ln_coeff_lte.copy().astype('float32')
        layer = len(d_affine)
        d_ineq_lte, d_ineq_gte = back_propagate_GPU1(d_ln_coeff_lte, d_ln_coeff_gte, d_affine, d_relu, layer,
                                                          if_activation)
        d_lbs_low, d_ubs_low = get_bounds_GPU(d_ineq_lte, d_ineq_lte, d_l1_lb, d_l1_ub)
        lbs_low = cp.asnumpy(d_lbs_low)
        flag = True
        #print(f"DEBUG OUTCOME Node{out1} --> lbs:{d_lbs}; ubs:{d_ubs}")
        for init_id in range(len(d_relu)):
            if(outcomes[init_id] == None):
                flag = True
                for out2 in outNodes:
                     if (out2 != out1) and (lbs_low[init_id][out2]<=0.0):
                        flag = False
                        break;
                if(flag == True):
                    stmt = inv_var_index[(len(d_affine)-1,out1)]
                    outcomes[init_id] = stmt
    if (print_mode == 1 or print_mode == 2):
        for init_id in range(len(outcomes)):
            # if (init_id not in [5348,5349,5364,5365]):
            #    continue
            print(
                f"init_id-> {init_id}; Ranges -> {commons.convertBound(d_l1_lb[init_id], d_l1_ub[init_id], inv_var_index, sensitive)}\nOut-> {outcomes[init_id]}")

    return outcomes

def active_convert(active_status,dims,inv_var_index,outcomes):
    activated = []
    deactivated = []
    for init_id in range(len(active_status)):
        node_num = 3
        act = []
        deact = []
        if (outcomes[init_id] != None):
            activated.append(act)
            deactivated.append(deact)
            continue
        for layer_index in range(1,len(dims[1:])):
            for neuron_index in range(1,dims[layer_index]):
                if(active_status[init_id,layer_index,neuron_index] == 0):
                    stmt = inv_var_index[(layer_index,neuron_index)]
                    deact.append(stmt)
                elif(active_status[init_id,layer_index,neuron_index] == 1):
                    stmt = inv_var_index[(layer_index,neuron_index)]
                    act.append(stmt)
                node_num+=1
            node_num+=1
        act = tuple(act)
        deact = tuple(deact)
        activated.append(act)
        deactivated.append(deact)
    return activated,deactivated

def detailedPrintCondense(d_affine,d_relu,d_active_pattern,d_l1_lb,d_l1_ub,if_activation,d_if_activation,var_index,inv_var_index,dims):
    l1_ub = cp.asnumpy(d_l1_ub)
    l1_lb = cp.asnumpy(d_l1_lb)
    init_id = 0 # len(d_l1_ub) -1
    print(f"init_id-> {init_id}; lbs -> {d_l1_lb[init_id]}; ubs -> {d_l1_ub[init_id]}")
    print(f"var_index = {var_index}")
    print(f"inv_var_index = {inv_var_index}")
    for i in range(1, len(d_affine)):
        d_ineq_lte, d_ineq_gte = back_propagate_GPU(d_affine, d_relu, i, if_activation)
        if (any(if_activation[i])):
            d_lbs_low, d_ubs_low = get_bounds_GPU(d_ineq_lte, d_ineq_lte, d_l1_lb, d_l1_ub)
            d_lbs_up, d_ubs_up = get_bounds_GPU(d_ineq_gte, d_ineq_gte, d_l1_lb, d_l1_ub)
            relu_compute_GPU(d_lbs_low, d_ubs_low, d_lbs_up, d_ubs_up, d_relu[:, i], d_active_pattern[:, i, :],
                             d_l1_lb, d_l1_ub,d_if_activation[i])
            relu = cp.asnumpy(d_relu)
        ineq_lte = cp.asnumpy(d_ineq_lte)
        ineq_gte = cp.asnumpy(d_ineq_gte)
        print(f"\t\t LAYER {i} Substituted")
        for j in range(1, dims[i]):
            print(f"\tNode {j}")
            print(f" eq LTE L1: {commons.ineq_str(ineq_lte[init_id][j], i, j, '>=', 0, inv_var_index)}")
            print(f" eq GTE L1: {commons.ineq_str(ineq_gte[init_id][j], i, j, '<=', 0, inv_var_index)}")
            print(f" eq LOW (LB,UB): {commons.get_bounds_single(ineq_lte[init_id], ineq_lte[init_id], j, l1_lb[init_id], l1_ub[init_id])}")
            print(f" eq UP (LB,UB): {commons.get_bounds_single(ineq_gte[init_id], ineq_gte[init_id], j, l1_lb[init_id], l1_ub[init_id])}")
        if (any(if_activation[i])):  # assuming if first node in a layer has activation then all do
            print(f"\t RELU-LAYER {i}")
            for j in range(1, dims[i]):
                print(f"\tNode {j}")
                print(f" Relu eq LTE: Slope: {relu[init_id][i][j][0]}, Y-Coeff: {relu[init_id][i][j][1]}")
                print(f" Relu eq GTE: Slope: {relu[init_id][i][j][2]}, Y-Coeff: {relu[init_id][i][j][3]}")
                relu_val = [relu[init_id][i][j][0], relu[init_id][i][j][1], relu[init_id][i][j][0],relu[init_id][i][j][1]]
                print(f"Relu{i}{j} eq LOW (LB,UB): {commons.get_bounds_single(ineq_lte[init_id], ineq_lte[init_id], j, l1_lb[init_id], l1_ub[init_id], relu_val=relu_val)}")
                relu_val = [relu[init_id][i][j][2], relu[init_id][i][j][3], relu[init_id][i][j][2],relu[init_id][i][j][3]]
                print(f"Relu{i}{j} eq UP (LB,UB): {commons.get_bounds_single(ineq_gte[init_id], ineq_gte[init_id], j, l1_lb[init_id], l1_ub[init_id], relu_val=relu_val)}")
        # print stuff
        else:
            print(f"\t\t NO RELU ON LAYER {i}")
    #print(f"activation->{d_active_pattern}")

def miniPrintCondense( d_affine, d_relu, d_active_pattern, d_l1_lb, d_l1_ub, if_activation,d_if_activation,inv_var_index,sensitive,dims):
    l1_ub = cp.asnumpy(d_l1_ub)
    l1_lb = cp.asnumpy(d_l1_lb)
    #init_id = 0  # len(d_l1_ub) -1
    #print(f"init_id-> {init_id}; lbs -> {l1_lb[init_id]}; ubs -> {l1_ub[init_id]}")
    for i in range(1, len(d_affine)):
        d_ineq_lte, d_ineq_gte = back_propagate_GPU(d_affine, d_relu, i, if_activation)
        if (any(if_activation[i])):
            d_lbs_low, d_ubs_low = get_bounds_GPU(d_ineq_lte, d_ineq_lte, d_l1_lb, d_l1_ub)
            d_lbs_up, d_ubs_up = get_bounds_GPU(d_ineq_gte, d_ineq_gte, d_l1_lb, d_l1_ub)
            relu_compute_GPU(d_lbs_low, d_ubs_low, d_lbs_up, d_ubs_up, d_relu[:,i], d_active_pattern[:, i, :],
                             d_l1_lb, d_l1_ub,d_if_activation[i])
            relu = cp.asnumpy(d_relu)
        ineq_lte = cp.asnumpy(d_ineq_lte)
        ineq_gte = cp.asnumpy(d_ineq_gte)

        for init_id in range(len(ineq_lte)):
            print(f"init_id-> {init_id}; Ranges -> {commons.convertBound(d_l1_lb[init_id], d_l1_ub[init_id], inv_var_index, sensitive)}")

            if (any(if_activation[i])):  # assuming if first node in a layer has activation then all do
                for j in range(1, dims[i]):
                    relu_val = [relu[init_id][i][j][0], relu[init_id][i][j][1], relu[init_id][i][j][0], relu[init_id][i][j][1]]
                    print(f"Relu {inv_var_index[(i,j)]} eq LOW (LB,UB): {commons.get_bounds_single(ineq_lte[init_id], ineq_lte[init_id], j, l1_lb[init_id], l1_ub[init_id], relu_val=relu_val)}")
                    relu_val = [relu[init_id][i][j][2], relu[init_id][i][j][3], relu[init_id][i][j][2], relu[init_id][i][j][3]]
                    print(f"Relu {inv_var_index[(i,j)]} eq UP (LB,UB): {commons.get_bounds_single(ineq_gte[init_id], ineq_gte[init_id], j, l1_lb[init_id], l1_ub[init_id], relu_val=relu_val)}")
            else:
                for j in range(1, dims[i]):
                    print(f" eq {inv_var_index[(i,j)]} LOW (LB,UB): {commons.get_bounds_single(ineq_lte[init_id], ineq_lte[init_id], j, l1_lb[init_id], l1_ub[init_id])}")
                    print(f" eq {inv_var_index[(i,j)]} UP (LB,UB): {commons.get_bounds_single(ineq_gte[init_id], ineq_gte[init_id], j, l1_lb[init_id], l1_ub[init_id])}")

def noPrintCondense(d_affine, d_relu, i, if_activation,d_if_activation,d_active_pattern, d_l1_lb,d_l1_ub):
    for i in range(1, len(d_affine)):
        d_ineq_lte, d_ineq_gte = back_propagate_GPU(d_affine, d_relu, i, if_activation)
        if (any(if_activation[i])):
            d_lbs_low, d_ubs_low = get_bounds_GPU(d_ineq_lte, d_ineq_lte, d_l1_lb, d_l1_ub)
            d_lbs_up, d_ubs_up = get_bounds_GPU(d_ineq_gte, d_ineq_gte, d_l1_lb, d_l1_ub)
            relu_compute_GPU(d_lbs_low, d_ubs_low,d_lbs_up, d_ubs_up, d_relu[:,i], d_active_pattern[:,i,:],d_l1_lb,d_l1_ub,d_if_activation[i])

def analyze(netGPU,l1_lbL,l1_ubL,percent,L,print_mode):
    d_affine,if_activation,d_if_activation,var_index,inv_var_index,outNodes,dims,l1_lb, l1_ub,sensitive,max_diff,NO_OF_LAYERS,MAX_NODES_IN_LAYER = netGPU
    if (l1_lbL == None):
        l1_lbL = l1_lb
        l1_ubL = l1_ub

    l1_lb_list, l1_ub_list = commons.splitInitial(l1_lbL, l1_ubL, sensitive,L)
    s = ""
    for l1_lb in l1_lb_list:
        s += " + " + str(l1_lb.shape[0])
    print(Fore.CYAN + f"\t\t\t\tIntial batches: {s}", Style.RESET_ALL)
    activatedL, deactivatedL, outcomeL = [], [], []
    for i in range(len(l1_ub_list)):
        if (len(l1_ub_list[i]) == 0):
            continue
        d_l1_lb = cp.asarray(l1_lb_list[i])
        d_l1_ub = cp.asarray(l1_ub_list[i])
        NO_OF_INITIALS = len(d_l1_lb)
        print(Fore.CYAN + f"\t\tNO_OF_LAYER:{NO_OF_LAYERS}; MAX_NODES_IN_LAYER:{MAX_NODES_IN_LAYER}; NO_OF_INITIALS:{NO_OF_INITIALS}",Style.RESET_ALL)
        # The 4 in relu is for lessThan(slope,y-coeff);greaterThan(slope,y-coeff)
        d_relu = cp.zeros((NO_OF_INITIALS, NO_OF_LAYERS + 1, MAX_NODES_IN_LAYER + 1, 4))
        d_active_pattern = cp.zeros((NO_OF_INITIALS, NO_OF_LAYERS + 1, MAX_NODES_IN_LAYER + 1)).astype(np.float32)

        # Removes NumbaPerformanceWarning and others but slow down everything significantly.
        warnings.filterwarnings("ignore")
        if (print_mode == 1):
            detailedPrintCondense(d_affine,d_relu,d_active_pattern,d_l1_lb,d_l1_ub,if_activation,d_if_activation,var_index,inv_var_index,dims)
        elif (print_mode == 2):
            miniPrintCondense( d_affine, d_relu, d_active_pattern, d_l1_lb, d_l1_ub, if_activation,d_if_activation,inv_var_index,sensitive,dims)
        elif (print_mode == 3):
            noPrintCondense(d_affine, d_relu, i, if_activation,d_if_activation, d_active_pattern, d_l1_lb, d_l1_ub)

        outcome = oneOutput(d_affine, d_relu, if_activation, d_l1_lb, d_l1_ub, outNodes, inv_var_index,sensitive,print_mode)
        active_pattern = cp.asnumpy(d_active_pattern)
        activated, deactivated = active_convert(active_pattern, dims, inv_var_index, outcome)
        activatedL.append(activated)
        deactivatedL.append(deactivated)
        outcomeL.append(outcome)
    factor = (len(l1_ub_list) - 1) * len(l1_ub_list[0]) + len(l1_ub_list[-1])
    print(f"factor:{(factor / len(l1_lbL))}")
    percent = percent / (factor / len(l1_lbL))
    return activatedL, deactivatedL, outcomeL, l1_lb_list, l1_ub_list, percent, inv_var_index
