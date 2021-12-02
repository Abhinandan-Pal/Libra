import time
import numpy as np
import cupy as cp
from numba import cuda
def get_bounds_GPU(d_l1_lte, d_l1_gte, l1_lb=-1, l1_ub=1):
    @cuda.jit
    def bound_helper(l1_lte, l1_gte, l1_lb, l1_ub, lbs, ubs):
        id = cuda.grid(1)
        if (id >= len(lbs)):
            return
        lbs[id] = l1_lte[id, 0]
        for coeff in l1_lte[id, 1:]:
            if (coeff < 0):
                lbs[id] += coeff * l1_ub[0]
            else:
                lbs[id] += coeff * l1_lb[0]
        ubs[id] = l1_gte[id, 0]
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


def relu_compute_GPU(d_l1_lte, d_l1_gte, d_relu_layer, d_active_pattern):
    @cuda.jit
    def relu_compute_helper(lbs, ubs, relu_layer, active_pattern):
        id = cuda.grid(1)
        if (id >= len(ubs)):
            return
        if (ubs[id] < 0):
            relu_layer[id] = (0.0, 0.0, 0.0, 0.0)
            active_pattern[id] = 0
            return  # as initialized with zeros
        if (lbs[id] > 0):
            relu_layer[id] = (1.0, 0.0, 1.0, 0.0)
            active_pattern[id] = 1
            return
        active_pattern[id] = 2
        slope = ubs[id] / (ubs[id] - lbs[id])
        y_coeff = -ubs[id] * lbs[id] / (ubs[id] - lbs[id])
        relu_layer[id] = (0.0, 0.0, slope, y_coeff)
        b3_area = abs(ubs[id] * (ubs[id] - lbs[id]))
        c3_area = abs(lbs[id] * (ubs[id] - lbs[id]))
        if (c3_area < b3_area):
            relu_layer[id] = (1.0, 0.0, slope, y_coeff)

    d_lbs, d_ubs = get_bounds_GPU(d_l1_lte, d_l1_gte)
    tpb = (min(1024, len(d_l1_lte)),)
    bpg = (int(np.ceil(len(d_l1_lte) / tpb[0])),)
    relu_compute_helper[bpg, tpb](d_lbs,
                                  d_ubs,
                                  d_relu_layer,
                                  d_active_pattern)
    return d_relu_layer, d_active_pattern


def back_propagate_GPU(affine, relu, layer: int, if_activation, active_pattern):
    # shift the CP creation to caller.
    d_affine = cp.asarray(affine)
    d_relu = cp.asarray(relu)
    d_active_pattern = cp.asarray(active_pattern)
    d_ln_coeff_lte = d_affine[layer].copy()
    d_ln_coeff_gte = d_affine[layer].copy()

    @cuda.jit
    def back_affine_helper(i, l1_lte, l1_gte, ln_coeff_lte, ln_coeff_gte, ineq_prev_lte, ineq_prev_gte):
        k, p = cuda.grid(2)
        i = i[0]
        if (k >= len(l1_lte) or p >= len(ineq_prev_lte[i])):
            return
        if (ln_coeff_lte[k][i] > 0):
            l1_lte[k][p] += ln_coeff_lte[k][i] * ineq_prev_lte[i][p]  # should it be i or i-1?
        else:
            l1_lte[k][p] += ln_coeff_lte[k][i] * ineq_prev_gte[i][p]
        if (ln_coeff_gte[k][i] > 0):
            l1_gte[k][p] += ln_coeff_gte[k][i] * ineq_prev_gte[i][p]
        else:
            l1_gte[k][p] += ln_coeff_gte[k][i] * ineq_prev_lte[i][p]

    def back_affine_GPU(d_ineq_prev_lte, d_ineq_prev_gte, d_ln_coeff_lte, d_ln_coeff_gte):
        d_l1_lte = cp.zeros(affine[layer].shape)
        d_l1_gte = cp.zeros(affine[layer].shape)
        d_l1_lte[:, 0] = d_ln_coeff_lte[:, 0]  # This doesnt work directly on few versions of cupy thus using numpy
        d_l1_gte[:, 0] = d_ln_coeff_gte[:, 0]

        cuda_iters = (len(d_l1_lte), len(d_ineq_prev_lte[1]))
        tpb = (min(32, cuda_iters[0]), min(32, cuda_iters[1]))
        bpg = (int(np.ceil(cuda_iters[0] / tpb[0])), int(np.ceil(cuda_iters[1] / tpb[1])))

        for i in range(1, len(d_l1_lte)):
            d_i = cp.array([i])
            # d_coeff_lte = cp.array(d_ln_coeff_lte[:, i])
            # d_coeff_gte = cp.array(d_ln_coeff_gte[:, i])
            # d_ineq_prev_lte_i = d_ineq_prev_lte[i]
            # d_ineq_prev_gte_i = d_ineq_prev_gte[i]

            back_affine_helper[bpg, tpb](d_i, d_l1_lte,
                                         d_l1_gte,
                                         d_ln_coeff_lte,
                                         d_ln_coeff_gte,
                                         d_ineq_prev_lte,
                                         d_ineq_prev_gte)
        return d_l1_lte, d_l1_gte

    @cuda.jit
    def back_relu_coeff_helper(relu_layer, ln_coeff_lte, ln_coeff_gte):
        i, j = cuda.grid(2)
        if (i < 1 or j < 1 or i >= len(ln_coeff_lte) or j >= len(relu_layer)):
            return
        if (ln_coeff_lte[i][j] > 0):
            ln_coeff_lte[i][j] = relu_layer[j][0] * ln_coeff_lte[i][j]
        else:
            ln_coeff_lte[i][j] = relu_layer[j][2] * ln_coeff_lte[i][j]
        if (ln_coeff_gte[i][j] > 0):
            ln_coeff_gte[i][j] = relu_layer[j][2] * ln_coeff_gte[i][j]
        else:
            ln_coeff_gte[i][j] = relu_layer[j][0] * ln_coeff_gte[i][j]

    @cuda.jit
    def back_relu_base_helper(relu_layer, ln_coeff_lte, ln_coeff_gte):
        i = cuda.grid(1)
        if (i < 1 or i >= len(ln_coeff_lte)):
            return
        for j in range(1, len(relu_layer)):
            if (ln_coeff_lte[i][j] > 0):
                ln_coeff_lte[i][0] += relu_layer[j][1] * ln_coeff_lte[i][j]
            else:
                ln_coeff_lte[i][0] += relu_layer[j][3] * ln_coeff_lte[i][j]
            if (ln_coeff_gte[i][j] > 0):
                ln_coeff_gte[i][0] += relu_layer[j][3] * ln_coeff_gte[i][j]
            else:
                ln_coeff_gte[i][0] += relu_layer[j][1] * ln_coeff_gte[i][j]

    def back_relu_GPU(d_relu_layer, d_ln_coeff_lte, d_ln_coeff_gte):
        cuda_iters1 = (len(d_ln_coeff_lte),)
        tpb1 = (min(1024, cuda_iters1[0]),)
        bpg1 = (int(np.ceil(cuda_iters1[0] / tpb1[0])),)
        back_relu_base_helper[bpg1, tpb1](d_relu_layer,
                                          d_ln_coeff_lte,
                                          d_ln_coeff_gte)

        cuda_iters = (len(d_ln_coeff_lte), len(d_relu_layer))
        tpb = (min(32, cuda_iters[0]), min(32, cuda_iters[1]))
        bpg = (int(np.ceil(cuda_iters[0] / tpb[0])), int(np.ceil(cuda_iters[1] / tpb[1])))
        back_relu_coeff_helper[bpg, tpb](d_relu_layer,
                                         d_ln_coeff_lte,
                                         d_ln_coeff_gte)

    layer_t = layer
    while (layer != 1):  # layer zero is input and layer one is in already in terms of input
        # First relu of previous layer
        if (if_activation[layer - 1][1] == True):
            # print(f"DEBUG--->relu{layer-1}:{relu[layer-1]}; ln_gte={ln_coeff_gte}")
            back_relu_GPU(d_relu[layer - 1], d_ln_coeff_lte, d_ln_coeff_gte)
        # print(f"DEBUG---> ln_lte AFTER ={ln_coeff_gte}")
        '''for j in range(1, len(affine[0])):
            print(f"\tNode {j}")
            print(f" eq RELU LTE L{layer}: {ineq_str(ln_coeff_lte[j], layer_t, j, '>=', layer-2)}")
            print(f" eq RELU GTE L{layer}: {ineq_str(ln_coeff_lte[j], layer_t, j, '<=', layer-2)}")
        '''
        # Then affine of previous layer
        d_ineq_prev_gte = d_affine[layer - 1]
        d_ineq_prev_lte = d_affine[layer - 1]
        d_ln_coeff_lte, d_ln_coeff_gte = back_affine_GPU(d_ineq_prev_lte, d_ineq_prev_gte, d_ln_coeff_lte,
                                                         d_ln_coeff_gte)
        '''
        for j in range(1, len(affine[0])):
            print(f"\tNode {j}")
            print(f" eq LTE L{layer}: {ineq_str(ln_coeff_lte[j], layer_t, j, '>=', layer-2)}")
            print(f" eq GTE L{layer}: {ineq_str(ln_coeff_lte[j], layer_t, j, '<=', layer-2)}")
        '''
        layer -= 1
    if (if_activation[layer_t][1] == 1):
        # print("DEBUG --> PERFORM RELU")
        relu_compute_GPU(d_ln_coeff_lte, d_ln_coeff_gte, d_relu[layer_t], d_active_pattern[layer_t])
    # print(f"DEBUG RELU layer: {layer_t} ---> {relu[layer_t]}")
    else:
        pass
    # print("DEBUG --> DONT PERFORM RELU")
    # return active_pattern
    # Different return for debug purposes
    return d_ln_coeff_lte, d_ln_coeff_gte


def get_bounds_CPU2(l1_lte, l1_gte, l1_lb=-1, l1_ub=1):
    lbs = np.zeros(l1_lte.shape[0])
    ubs = np.zeros(l1_lte.shape[0])
    for i in range(len(l1_lte)):
        lbs[i] = l1_lte[i][0]
        for coeff in l1_lte[i][1:]:
            if (coeff < 0):
                lbs[i] += coeff * l1_ub
            else:
                lbs[i] += coeff * l1_lb
        ubs[i] = l1_gte[i][0]
        for coeff in l1_gte[i][1:]:
            if (coeff > 0):
                ubs[i] += coeff * l1_ub
            else:
                ubs[i] += coeff * l1_lb
    return lbs, ubs


def relu_compute_CPU(l1_lte, l1_gte, relu_layer, active_pattern):
    lbs, ubs = get_bounds_CPU2(l1_lte, l1_gte)
    relu_layer = np.zeros((len(l1_lte), 4))
    active_pattern = np.zeros(l1_lte.shape[0])
    for i in range(len(l1_lte)):
        if (ubs[i] < 0):
            relu_layer[i] = [0, 0, 0, 0]
            active_pattern[i] = 0
            continue
        elif (lbs[i] > 0):
            relu_layer[i] = [1, 0, 1, 0]
            active_pattern[i] = 1
        else:
            active_pattern[i] = 2
            # print(f"DEBUG ----> ubs: {ubs[i]};lbs {lbs[i]}")
            slope = ubs[i] / (ubs[i] - lbs[i])
            y_coeff = -ubs[i] * lbs[i] / (ubs[i] - lbs[i])
            # print(f"DEBUG ----> slope: {slope};y_coeff {y_coeff}")
            relu_layer[i] = [0, 0, slope, y_coeff]
            b3_area = abs(ubs[i] * (ubs[i] - lbs[i]))
            c3_area = abs(lbs[i] * (ubs[i] - lbs[i]))
            if (c3_area < b3_area):
                relu_layer[i] = [1, 0, slope, y_coeff]


def back_propagate(affine, relu, layer: int, if_activation, active_pattern):
    ln_coeff_lte = affine[layer].copy()
    ln_coeff_gte = affine[layer].copy()

    def back_affine(ineq_prev_lte, ineq_prev_gte, ln_coeff_lte, ln_coeff_gte):
        l1_lte = np.zeros(affine[layer].shape)
        l1_gte = np.zeros(affine[layer].shape)
        l1_lte[:, 0] = ln_coeff_lte[:, 0]
        l1_gte[:, 0] = ln_coeff_gte[:, 0]
        for i in range(1,
                       len(l1_lte)):  # loop through each coeff of a node of current layer and nodes of previous layer
            for k in range(0, len(l1_lte)):  # loop through all nodes
                for p in range(0, len(ineq_prev_lte[1])):  # loop thorugh coeffients of a node of previous layer.
                    if (ln_coeff_lte[k][i] > 0):
                        l1_lte[k][p] += ln_coeff_lte[k][i] * ineq_prev_lte[i][p]  # should it be i or i-1?
                    else:
                        l1_lte[k][p] += ln_coeff_lte[k][i] * ineq_prev_gte[i][p]
                    if (ln_coeff_gte[k][i] > 0):
                        l1_gte[k][p] += ln_coeff_gte[k][i] * ineq_prev_gte[i][p]
                    else:
                        l1_gte[k][p] += ln_coeff_gte[k][i] * ineq_prev_lte[i][p]
        return l1_lte, l1_gte

    def back_relu(relu_layer, ln_coeff_lte, ln_coeff_gte):
        for i in range(1, len(ln_coeff_lte)):
            for j in range(1, len(relu_layer)):
                if (ln_coeff_lte[i][j] > 0):
                    # t = ln_coeff_lte[i][j].copy()
                    ln_coeff_lte[i][0] += relu_layer[j][1] * ln_coeff_lte[i][j]
                    ln_coeff_lte[i][j] = relu_layer[j][0] * ln_coeff_lte[i][
                        j]  # [1:] to make sure base term is not affected
                else:
                    # t = ln_coeff_lte[i][j].copy()
                    ln_coeff_lte[i][0] += relu_layer[j][3] * ln_coeff_lte[i][j]
                    ln_coeff_lte[i][j] = relu_layer[j][2] * ln_coeff_lte[i][j]
                if (ln_coeff_gte[i][j] > 0):
                    # print(f"DEBUG REV ---> i:{i} ; j:{j} ; coeff: {ln_coeff_gte[i][j]}")
                    # t = ln_coeff_gte[i][j].copy()
                    ln_coeff_gte[i][0] += relu_layer[j][3] * ln_coeff_gte[i][j]
                    ln_coeff_gte[i][j] = relu_layer[j][2] * ln_coeff_gte[i][j]
                else:
                    # print(f"DEBUG REV ---> i:{i} ; j:{j} ; coeff: {ln_coeff_gte[i][j]}")
                    # t = ln_coeff_gte[i][j].copy()
                    ln_coeff_gte[i][0] += relu_layer[j][1] * ln_coeff_gte[i][j]
                    ln_coeff_gte[i][j] = relu_layer[j][0] * ln_coeff_gte[i][j]

    layer_t = layer
    while (layer != 1):  # layer zero is input and layer one is in already in terms of input
        # First relu of previous layer
        if (if_activation[layer - 1][1] == True):
            # print(f"DEBUG--->relu{layer-1}:{relu[layer-1]}; ln_gte={ln_coeff_gte}")
            back_relu(relu[layer - 1], ln_coeff_lte, ln_coeff_gte)
        # print(f"DEBUG---> ln_lte AFTER ={ln_coeff_gte}")
        '''for j in range(1, len(affine[0])):
            print(f"\tNode {j}")
            print(f" eq RELU LTE L{layer}: {ineq_str(ln_coeff_lte[j], layer_t, j, '>=', layer-2)}")
            print(f" eq RELU GTE L{layer}: {ineq_str(ln_coeff_lte[j], layer_t, j, '<=', layer-2)}")
        '''
        # Then affine of previous layer
        ineq_prev_gte = affine[layer - 1]
        ineq_prev_lte = affine[layer - 1]
        ln_coeff_lte, ln_coeff_gte = back_affine(ineq_prev_lte, ineq_prev_gte, ln_coeff_lte, ln_coeff_gte)
        '''
        for j in range(1, len(affine[0])):
            print(f"\tNode {j}")
            print(f" eq LTE L{layer}: {ineq_str(ln_coeff_lte[j], layer_t, j, '>=', layer-2)}")
            print(f" eq GTE L{layer}: {ineq_str(ln_coeff_lte[j], layer_t, j, '<=', layer-2)}")
        '''
        layer -= 1
    if (if_activation[layer_t][1] == 1):
        # print("DEBUG --> PERFORM RELU")
        relu_compute_CPU(ln_coeff_lte, ln_coeff_gte, relu[layer_t], active_pattern[layer_t])
    # print(f"DEBUG RELU layer: {layer_t} ---> {relu[layer_t]}")
    else:
        pass
    # print("DEBUG --> DONT PERFORM RELU")
    # return active_pattern

    # Different return for debug purposes
    return ln_coeff_lte, ln_coeff_gte


def caller3():
    MAX_NODES_IN_LAYER = 200
    NO_OF_LAYERS = 10

    affine = np.random.randn(NO_OF_LAYERS + 1, MAX_NODES_IN_LAYER + 1, MAX_NODES_IN_LAYER + 1).astype(np.float32)
    relu = np.random.randn(NO_OF_LAYERS + 1, MAX_NODES_IN_LAYER + 1, 4).astype(np.float32)
    if_activation = np.ones((NO_OF_LAYERS + 1, MAX_NODES_IN_LAYER + 1))
    active_pattern = np.zeros((NO_OF_LAYERS + 1, MAX_NODES_IN_LAYER + 1))
    layer = 2
    # print(f"Affine -> {affine}")

    print(f"MAX_NODES_IN_LAYER = {MAX_NODES_IN_LAYER}")
    time_sec = time.time()
    print(f"{back_propagate(affine, relu, layer, if_activation, active_pattern)[0][0][0]}")
    time_sec = time.time() - time_sec
    print(f"CPU time: {time_sec}\n\n")

    time_sec = time.time()
    print(f"{back_propagate_GPU(affine, relu, layer, if_activation, active_pattern)[0][0][0]}")
    time_sec = time.time() - time_sec
    print(f"GPU time: {time_sec}")


caller3()