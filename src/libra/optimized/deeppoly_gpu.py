import random
import math
import numpy as np
import cupy as cp
from numba import cuda
from libra.optimized import commons
from libra.optimized.commons import texpr_to_dict, get_bounds_single, ineq_str, ineq_str_direct
from libra.core.cfg import Node, Function, Activation, Basic
from libra.core.expressions import VariableIdentifier
import warnings
from apronpy.var import PyVar
import copy
import time
from itertools import product


def get_bounds_GPU(d_l1_lte, d_l1_gte, d_l1_lb, d_l1_ub):
    @cuda.jit
    def bound_helper(l1_lte, l1_gte, l1_lb, l1_ub, lbs, ubs):
        init_id, id = cuda.grid(2)
        if (id >= len(lbs[0]) or init_id >= len(lbs)):
            return
        lbs[init_id][id] = l1_lte[init_id, id, 0]
        for i in range(1, len(l1_lte[0][id])):
            if (l1_lte[init_id, id, i] < 0):
                lbs[init_id][id] += l1_lte[init_id, id, i] * l1_ub[init_id][i]
            else:
                lbs[init_id][id] += l1_lte[init_id, id, i] * l1_lb[init_id][i]
        ubs[init_id][id] = l1_gte[init_id, id, 0]
        for i in range(1, len(l1_gte[0][id])):
            if (l1_gte[init_id, id, i] > 0):
                ubs[init_id][id] += l1_gte[init_id, id, i] * l1_ub[init_id][i]
            else:
                ubs[init_id][id] += l1_gte[init_id, id, i] * l1_lb[init_id][i]

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


def relu_compute_GPU(d_lbs, d_ubs, d_relu_layer, d_active_pattern, d_f_act_pattern):
    @cuda.jit
    def relu_compute_helper(lbs, ubs, relu_layer, active_pattern, f_act_pattern):
        init_id, id = cuda.grid(2)
        if (id < 1 or id >= len(ubs[0]) or init_id >= len(ubs)):
            return
        if (ubs[init_id][id] <= 0 or f_act_pattern[init_id][id] == 1):
            relu_layer[init_id][id] = (0.0, 0.0, 0.0, 0.0)
            active_pattern[init_id][id] = 0
            return  # as initialized with zeros
        if (lbs[init_id][id] >= 0 or f_act_pattern[init_id][id] == 2):
            relu_layer[init_id][id] = (1.0, 0.0, 1.0, 0.0)
            active_pattern[init_id][id] = 1
            return
        active_pattern[init_id][id] = 2
        slope = ubs[init_id][id] / (ubs[init_id][id] - lbs[init_id][id])
        y_coeff = -ubs[init_id][id] * lbs[init_id][id] / (ubs[init_id][id] - lbs[init_id][id])
        relu_layer[init_id][id] = (0.0, 0.0, slope, y_coeff)
        b3_area = abs(ubs[init_id][id] * (ubs[init_id][id] - lbs[init_id][id]))
        c3_area = abs(lbs[init_id][id] * (ubs[init_id][id] - lbs[init_id][id]))
        if (c3_area < b3_area):
            relu_layer[init_id][id] = (1.0, 0.0, slope, y_coeff)

    cuda_iters1 = (len(d_lbs), len(d_lbs[0]))
    tpb1 = (min(64, cuda_iters1[0]), min(16, cuda_iters1[1]))
    bpg1 = (int(np.ceil(cuda_iters1[0] / tpb1[0])), int(np.ceil(cuda_iters1[1] / tpb1[1])))
    relu_compute_helper[bpg1, tpb1](d_lbs,
                                    d_ubs,
                                    d_relu_layer,
                                    d_active_pattern,
                                    d_f_act_pattern)
    return d_relu_layer, d_active_pattern


@cuda.jit
def back_affine_helper(i, l1_lte, l1_gte, ln_coeff_lte, ln_coeff_gte, ineq_prev_lte, ineq_prev_gte):
    init_id, k, p = cuda.grid(3)
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

    cuda_iters = (len(d_l1_lte), len(d_l1_lte[1]), len(d_ineq_prev_lte[1]))
    tpb = (min(16, cuda_iters[0]), min(8, cuda_iters[1]), min(8, cuda_iters[2]))
    bpg = (
    int(np.ceil(cuda_iters[0] / tpb[0])), int(np.ceil(cuda_iters[1] / tpb[1])), int(np.ceil(cuda_iters[2] / tpb[2])))

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
    init_id, i, j = cuda.grid(3)
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
    init_id, i = cuda.grid(2)
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


def back_propagate_GPU1(d_ln_coeff_lte, d_ln_coeff_gte, d_affine, d_relu, layer: int, if_activation):
    while (layer != 1):  # layer zero is input and layer one is in already in terms of input
        # First relu of previous layer
        if (if_activation[layer - 1][1] == True):
            back_relu_GPU(d_relu[:, layer - 1], d_ln_coeff_lte, d_ln_coeff_gte)
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
    d_ln_coeff_lte, d_ln_coeff_gte = back_propagate_GPU1(d_ln_coeff_lte, d_ln_coeff_gte, d_affine, d_relu, layer,
                                                         if_activation)
    return d_ln_coeff_lte, d_ln_coeff_gte


def oneOutput(d_affine, d_relu, if_activation, d_l1_lb, d_l1_ub, outNodes, inv_var_index):
    outcomes = [None] * len(d_relu)
    for out1 in outNodes:
        ln_shape = (len(d_relu),) + d_affine[0].shape
        ln_coeff_lte = np.zeros(ln_shape).astype('float32')
        for out2 in outNodes:
            if (out2 != out1):
                ln_coeff_lte[:, out2, out1] = 1.0
                ln_coeff_lte[:, out2, out2] = -1.0
        d_ln_coeff_lte = cp.asarray(ln_coeff_lte)
        d_ln_coeff_gte = d_ln_coeff_lte.copy().astype('float32')
        layer = len(d_affine)
        while (layer != 1):  # layer zero is input and layer one is in already in terms of input
            # First relu of previous layer
            if (if_activation[layer - 1][1] == True):
                back_relu_GPU(d_relu[layer - 1], d_ln_coeff_lte, d_ln_coeff_gte)
            # Then affine of previous layer
            d_ineq_prev_gte = d_affine[layer - 1]
            d_ineq_prev_lte = d_affine[layer - 1]
            d_ln_coeff_lte, d_ln_coeff_gte = back_affine_GPU(d_ineq_prev_lte, d_ineq_prev_gte, d_ln_coeff_lte,
                                                             d_ln_coeff_gte)
            layer -= 1
        d_lbs, d_ubs = get_bounds_GPU(d_ln_coeff_lte, d_ln_coeff_gte, d_l1_lb, d_l1_ub)

        lbs = cp.asnumpy(d_lbs)
        # print(f"DEBUG OUTCOME Node{out1} --> lbs:{d_lbs}; ubs:{d_ubs}")
        for init_id in range(len(d_relu)):
            if (outcomes[init_id] == None):
                flag = True
                for out2 in outNodes:
                    if (out2 != out1) and (lbs[init_id][out2] <= 0.0):
                        flag = False
                        break;
                if (flag == True):
                    stmt = inv_var_index[(len(d_affine) - 1, out1)]
                    outcomes[init_id] = stmt
    return outcomes


def active_convert(active_status, dims, inv_var_index, outcomes):
    activated = []
    deactivated = []
    for init_id in range(len(active_status)):
        node_num = 3
        act = set()
        deact = set()
        if (outcomes[init_id] != None):
            activated.append(act)
            deactivated.append(deact)
            continue
        for layer_index in range(1, len(dims[1:])):
            for neuron_index in range(1, dims[layer_index]):
                if (active_status[init_id, layer_index, neuron_index] == 0):
                    stmt = inv_var_index[(layer_index, neuron_index)]
                    val = Basic(node_num, [PyVar(stmt)])
                    deact.add(val)
                elif (active_status[init_id, layer_index, neuron_index] == 1):
                    stmt = inv_var_index[(layer_index, neuron_index)]
                    val = Basic(node_num, [PyVar(stmt)])
                    act.add(val)
                node_num += 1
            node_num += 1
        activated.append(act)
        deactivated.append(deact)
    return activated, deactivated


def detailedPrintCondense(d_affine, d_relu, d_active_pattern, d_l1_lb, d_l1_ub, if_activation, relu, var_index,
                          inv_var_index, d_f_act_pattern):
    l1_ub = cp.asnumpy(d_l1_ub)
    l1_lb = cp.asnumpy(d_l1_lb)
    print(f"var_index = {var_index}")
    print(f"inv_var_index = {inv_var_index}")
    for i in range(1, len(d_affine)):
        print(f"\t\t LAYER {i} Input Equations")
        for j in range(1, len(d_affine[0])):
            print(
                f"Node: {j} -> if_activation: {if_activation[i, j]}\n eq: {ineq_str(d_affine[i, j], i, j, '=', i - 1, inv_var_index)} ")
        d_ineq_lte, d_ineq_gte = back_propagate_GPU(d_affine, d_relu, i, if_activation)
        if (if_activation[i][1] == 1):
            d_lbs, d_ubs = get_bounds_GPU(d_ineq_lte, d_ineq_gte, d_l1_lb, d_l1_ub)
            relu_compute_GPU(d_lbs, d_ubs, d_relu[:, i], d_active_pattern[:, i], d_f_act_pattern[:, i])
            relu = cp.asnumpy(d_relu)
        ineq_lte = cp.asnumpy(d_ineq_lte)
        ineq_gte = cp.asnumpy(d_ineq_gte)
        init_id = 1
        print(f"\t\t LAYER {i} Substituted")
        for j in range(1, len(d_affine[0])):
            print(f"\tNode {j}")
            print(f" eq LTE L1: {ineq_str(ineq_lte[init_id][j], i, j, '>=', 0, inv_var_index)}")
            print(f" eq GTE L1: {ineq_str(ineq_gte[init_id][j], i, j, '<=', 0, inv_var_index)}")
            print(
                f" eq (LB,UB): {get_bounds_single(ineq_lte[init_id], ineq_gte[init_id], j, l1_lb[init_id], l1_ub[init_id])}")
        if (if_activation[i, 1] == 1):  # assuming if first node in a layer has activation then all do
            print(f"\t RELU-LAYER {i}")
            for j in range(1, len(d_affine[0])):
                print(f"\tNode {j}")
                print(f" Relu eq LTE: Slope: {relu[init_id][i][j][0]}, Y-Coeff: {relu[init_id][i][j][1]}")
                print(f" Relu eq GTE: Slope: {relu[init_id][i][j][2]}, Y-Coeff: {relu[init_id][i][j][3]}")
                print(
                    f"Relu eq (LB,UB): {get_bounds_single(ineq_lte[init_id], ineq_gte[init_id], j, l1_lb[init_id], l1_ub[init_id], relu_val=relu[init_id][i][j])}")
        # print stuff
        else:
            print(f"\t\t NO RELU ON LAYER {i}")
    print(f"activation->{d_active_pattern}")


def miniPrintCondense(d_affine, d_relu, d_active_pattern, d_l1_lb, d_l1_ub, if_activation, d_f_act_pattern):
    l1_ub = cp.asnumpy(d_l1_ub)
    l1_lb = cp.asnumpy(d_l1_lb)
    init_id = 4  # len(d_l1_ub) -1
    print(f"init_id-> {init_id}; lbs -> {d_l1_lb[init_id]}; ubs -> {d_l1_ub[init_id]}")
    for i in range(1, len(d_affine)):
        d_ineq_lte, d_ineq_gte = back_propagate_GPU(d_affine, d_relu, i, if_activation)
        if (if_activation[i][1] == 1):
            d_lbs, d_ubs = get_bounds_GPU(d_ineq_lte, d_ineq_gte, d_l1_lb, d_l1_ub)
            relu_compute_GPU(d_lbs, d_ubs, d_relu[:, i], d_active_pattern[:, i], d_f_act_pattern[:, i])
            relu = cp.asnumpy(d_relu)

        ineq_lte = cp.asnumpy(d_ineq_lte)
        ineq_gte = cp.asnumpy(d_ineq_gte)

        if (if_activation[i, 1] == 1):  # assuming if first node in a layer has activation then all do
            for j in range(1, len(d_affine[0])):
                print(
                    f"Relu {i}:{j} eq (LB,UB): {get_bounds_single(ineq_lte[init_id], ineq_gte[init_id], j, l1_lb[init_id], l1_ub[init_id], relu_val=relu[init_id][i][j])}")
        else:
            for j in range(1, len(d_affine[0])):
                print(
                    f"{i}:{j} eq (LB,UB): {get_bounds_single(ineq_lte[init_id], ineq_gte[init_id], j, l1_lb[init_id], l1_ub[init_id])}")


def noPrintCondense(d_affine, d_relu, i, if_activation, d_active_pattern, d_l1_lb, d_l1_ub, d_f_act_pattern):
    for i in range(1, len(d_affine)):
        d_ineq_lte, d_ineq_gte = back_propagate_GPU(d_affine, d_relu, i, if_activation)  # place inside if and
        if (if_activation[i][1] == 1):  # TODO: adjust for forced activate.
            d_lbs, d_ubs = get_bounds_GPU(d_ineq_lte, d_ineq_gte, d_l1_lb, d_l1_ub)
            relu_compute_GPU(d_lbs, d_ubs, d_relu[:, i], d_active_pattern[:, i], d_f_act_pattern[:, i])


def network_condense_GPU(nodes, initial, forced_active, forced_inactive, outputs):
    L = 0.5
    U = 20
    # equation[n1][n2] stores the bias and coeff of nodes of previous layer to form x[n1][n2] in order
    # if_activation[n1][n2] stores if there is an activation on x[n1][n2] (only relu considered for now)
    NO_OF_LAYERS, MAX_NODES_IN_LAYER = commons.getNetShape(nodes)
    time_sec = time.time()
    l1_lb_list, l1_ub_list = commons.fillInitials(initial, L, MAX_NODES_IN_LAYER)
    time_sec = time.time() - time_sec
    s= ""
    for l1_lb in l1_lb_list:
        s+=" + " + str(l1_lb.shape[0])
    print(f"Intial batches: {s}")
    print(f"Fill Initial time: {time_sec}\n\n")
    for i in range(len(l1_ub_list)):
        d_l1_lb = cp.asarray(l1_lb_list[i])
        d_l1_ub = cp.asarray(l1_ub_list[i])

        NO_OF_INITIALS = len(d_l1_lb)
        print(f"NO_OF_LAYER:{NO_OF_LAYERS}; MAX_NODES_IN_LAYER:{MAX_NODES_IN_LAYER}; NO_OF_INITIALS:{NO_OF_INITIALS}")

        var_index: dict(str, (int, int)) = dict()

        affine = np.zeros((NO_OF_LAYERS + 1, MAX_NODES_IN_LAYER + 1, MAX_NODES_IN_LAYER + 1)).astype(np.float32)
        # relu = np.zeros((NO_OF_INITIALS,NO_OF_LAYERS + 1, MAX_NODES_IN_LAYER + 1, 4)).astype(np.float32)
        if_activation = np.zeros((NO_OF_LAYERS + 1, MAX_NODES_IN_LAYER + 1)).astype(np.int16)
        # active_pattern = np.zeros((NO_OF_INITIALS,NO_OF_LAYERS + 1, MAX_NODES_IN_LAYER + 1)).astype(np.int16)
        dims = np.ones(NO_OF_LAYERS + 1).astype(np.int32)
        f_act_pattern = np.zeros((NO_OF_INITIALS, NO_OF_LAYERS + 1, MAX_NODES_IN_LAYER + 1)).astype(np.int16)
        # obtain the lower bound and upper bound for input layer using "initial"
        # Assuming "initial" contains input from 0 to nth input in order.
        '''
            Convert forced_active, forced_inactive to f_act_pattern
        '''
        i = 1
        row_id, col_id, flag = (0, 1, False)
        for var, bound in initial.bounds.items():
            var_index[str(var)] = (row_id, col_id)
            col_id += 1

        '''for ini in range(NO_OF_INITIALS):
            i=1
            for var, bound in initial.bounds.items():
                a = random.uniform(bound.lower,bound.upper)
                b = random.uniform(bound.lower,bound.upper)
                l1_lb[ini][i] = min(a, b)  # bound.lower
                l1_ub[ini][i] = max(a, b)  # bound.upper
                i += 1'''
        commons.fillInput(nodes, affine, dims, if_activation, var_index, MAX_NODES_IN_LAYER)
        outNodes = set()
        for output in outputs:
            outNodes.add(var_index[str(output)][1])
        inv_var_index = {v: k for k, v in var_index.items()}

        # can remove layer 0 as it has no inequations
        # All these print are for debug mode. Actual will only activation pattern.
        d_affine = cp.asarray(affine)
        d_relu = cp.zeros((NO_OF_INITIALS, NO_OF_LAYERS + 1, MAX_NODES_IN_LAYER + 1, 4))
        d_active_pattern = cp.zeros((NO_OF_INITIALS, NO_OF_LAYERS + 1, MAX_NODES_IN_LAYER + 1))
        d_f_act_pattern = cp.asarray(f_act_pattern)

        # Removes NumbaPerformanceWarning and others but slow down everything significantly.
        warnings.filterwarnings("ignore")
        # detailedPrintCondense(d_affine,d_relu,d_active_pattern,d_l1_lb,d_l1_ub,if_activation,relu,var_index,inv_var_index,d_f_act_pattern)
        # miniPrintCondense(d_affine, d_relu, d_active_pattern, d_l1_lb, d_l1_ub, if_activation,d_f_act_pattern)
        noPrintCondense(d_affine, d_relu, i, if_activation, d_active_pattern, d_l1_lb, d_l1_ub, d_f_act_pattern)

        outcome = oneOutput(d_affine, d_relu, if_activation, d_l1_lb, d_l1_ub, outNodes, inv_var_index)
        active_pattern = cp.asnumpy(d_active_pattern)
        activated, deactivated = active_convert(active_pattern, dims, inv_var_index, outcome)
        '''for i in range(NO_OF_INITIALS):
            print(f"l1_lb -> {d_l1_lb[i]}; l1_ub -> {d_l1_ub[i]}")
            print(f"active_pattern: {active_pattern[i]}")
            print(f"GPU active:{activated[i]}; deactive:{deactivated[i]}; outcome:{outcome[i]}")'''
        # return activated, deactivated, outcome
