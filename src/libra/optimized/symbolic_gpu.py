import numpy as np
import cupy as cp
from numba import cuda
from libra.optimized.commons import texpr_to_dict, get_bounds_single, ineq_str
from libra.core.cfg import Node, Function, Activation,Basic
from libra.core.expressions import VariableIdentifier
import warnings
from apronpy.var import PyVar
import copy
import random


def get_bounds_GPU(d_l1_lte, d_l1_gte, d_l1_lb, d_l1_ub):
    @cuda.jit
    def bound_helper(l1_lte, l1_gte, l1_lb, l1_ub, lbs, ubs):
        init_id,id = cuda.grid(2)
        if (id >= len(lbs[0]) or init_id >= len(lbs)):
            return
        lbs[init_id][id] = l1_lte[init_id,id, 0]
        for i in range(1,len(l1_lte[0][id])):
            if(l1_lte[init_id,id,i]<0):
                lbs[init_id][id] += l1_lte[init_id,id,i] * l1_ub[init_id][i]
            else:
                lbs[init_id][id] += l1_lte[init_id,id,i] * l1_lb[init_id][i]
        ubs[init_id][id] = l1_gte[init_id,id, 0]
        for i in range(1,len(l1_gte[0][id])):
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


def relu_compute_GPU(d_lbs, d_ubs, d_symb_layer, d_active_pattern,d_l1_lb,d_l1_ub):
    @cuda.jit
    def relu_compute_helper(lbs, ubs, symb_layer, active_pattern):
        init_id,id = cuda.grid(2)
        if (id< 1 or id >= len(ubs[0]) or init_id >= len(ubs)):
            return
        if (ubs[init_id,id] <= 0):
            symb_layer[init_id,id] = (0.0, 0.0, 0.0)
            active_pattern[init_id,id] = 0
            return  # as initialized with zeros
        if (lbs[init_id,id] >= 0):
            symb_layer[init_id,id] = (0.0, 0.0, 0.0)
            active_pattern[init_id,id] = 1
            return
        active_pattern[init_id,id] = 2
        symb_layer[init_id,id] = (1.0, ubs[init_id,id], 0.0)

    cuda_iters1 = (len(d_lbs), len(d_lbs[0]))
    tpb1 = (min(64, cuda_iters1[0]), min(16, cuda_iters1[1]))
    bpg1 = (int(np.ceil(cuda_iters1[0] / tpb1[0])), int(np.ceil(cuda_iters1[1] / tpb1[1])))
    relu_compute_helper[bpg1, tpb1](d_lbs,
                                  d_ubs,
                                  d_symb_layer,
                                  d_active_pattern)
    return d_symb_layer, d_active_pattern

@cuda.jit
def back_affine_helper(i, l1_lte, l1_gte, ln_coeff_lte, ln_coeff_gte, ineq_prev_lte, ineq_prev_gte,symb_layer):
    init_id,k, p = cuda.grid(3)
    i = i[0]
    if (k >= len(l1_lte[0]) or p >= len(ineq_prev_lte[i]) or init_id >= len(l1_lte)):
        return
    if (symb_layer[init_id][i][0] == 0.0):
        if (ln_coeff_lte[init_id][k][i] > 0):
            l1_lte[init_id][k][p] += ln_coeff_lte[init_id][k][i] * ineq_prev_lte[i][p]
        else:
            l1_lte[init_id][k][p] += ln_coeff_lte[init_id][k][i] * ineq_prev_gte[i][p]
        if (ln_coeff_gte[init_id][k][i] > 0):
            l1_gte[init_id][k][p] += ln_coeff_gte[init_id][k][i] * ineq_prev_gte[i][p]
        else:
            l1_gte[init_id][k][p] += ln_coeff_gte[init_id][k][i] * ineq_prev_lte[i][p]

def back_affine_GPU(d_ineq_prev_lte, d_ineq_prev_gte, d_ln_coeff_lte, d_ln_coeff_gte,d_symb_layer):
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
                                     d_ineq_prev_gte,
                                     d_symb_layer)
    return d_l1_lte, d_l1_gte


@cuda.jit
def back_relu_helper(symb_layer, ln_coeff_lte, ln_coeff_gte):
    init_id,i = cuda.grid(2)
    if (i < 1 or i >= len(ln_coeff_lte[0]) or init_id >= len(ln_coeff_lte)):
        return
    for j in range(1, len(symb_layer[init_id])):
        if(symb_layer[init_id][j][0] == 1.0):
            if (ln_coeff_lte[init_id][i][j] > 0):
                ln_coeff_lte[init_id][i][0] += 0 * ln_coeff_lte[init_id][i][j]
            else:
                ln_coeff_lte[init_id][i][0] += symb_layer[init_id][j][1] * ln_coeff_lte[init_id][i][j]
            if (ln_coeff_gte[init_id][i][j] > 0):
                ln_coeff_gte[init_id][i][0] += symb_layer[init_id][j][1] * ln_coeff_gte[init_id][i][j]
            else:
                ln_coeff_gte[init_id][i][0] += 0 * ln_coeff_gte[init_id][i][j]

def back_relu_GPU(d_symb_layer, d_ln_coeff_lte, d_ln_coeff_gte):
    cuda_iters1 = (len(d_ln_coeff_lte),len(d_ln_coeff_lte[0]))
    tpb1 = (min(64, cuda_iters1[0]), min(16, cuda_iters1[1]))
    bpg1 = (int(np.ceil(cuda_iters1[0] / tpb1[0])), int(np.ceil(cuda_iters1[1] / tpb1[1])))
    back_relu_helper[bpg1, tpb1](d_symb_layer,
                                      d_ln_coeff_lte,
                                      d_ln_coeff_gte)



def back_propagate_GPU(d_affine, d_symb, layer: int, if_activation, d_active_pattern,d_l1_lb,d_l1_ub):
    # shift the CP creation to caller.
    ln_shape = (len(d_symb),) + d_affine[layer].shape
    ln_coeff_lte = np.zeros(ln_shape)
    ln_coeff_lte[:] = d_affine[layer].get()
    d_ln_coeff_lte = cp.asarray(ln_coeff_lte)

    ln_coeff_gte = np.zeros(ln_shape)
    ln_coeff_gte[:] = d_affine[layer].get()
    d_ln_coeff_gte = cp.asarray(ln_coeff_gte)
    layer_t = layer
    while (layer != 1):  # layer zero is input and layer one is in already in terms of input
        # First relu of previous layer
        if (if_activation[layer - 1][1] == True):
            back_relu_GPU(d_symb[:,layer - 1], d_ln_coeff_lte, d_ln_coeff_gte)
        # Then affine of previous layer
        d_ineq_prev_gte = d_affine[layer - 1]
        d_ineq_prev_lte = d_affine[layer - 1]
        d_ln_coeff_lte, d_ln_coeff_gte = back_affine_GPU(d_ineq_prev_lte, d_ineq_prev_gte, d_ln_coeff_lte,
                                                         d_ln_coeff_gte,d_symb[:,layer - 1])
        layer -= 1
    '''if (if_activation[layer_t][1] == 1):
        relu_compute_GPU(d_ln_coeff_lte, d_ln_coeff_gte, d_symb[layer_t], d_active_pattern[layer_t],d_l1_lb,d_l1_ub)
    else:
        pass'''
    #return d_active_pattern
    ''''# Different return for debug purposes'''
    #ln_coeff_gte = cp.asnumpy(d_ln_coeff_gte).astype(np.float32)
    #ln_coeff_lte = cp.asnumpy(d_ln_coeff_lte).astype(np.float32)
    return d_ln_coeff_lte, d_ln_coeff_gte

def oneOutput(last,d_affine,d_symb,if_activation,d_l1_lb,d_l1_ub,outNodes,inv_var_index):
    outcomes = [None] * len(d_symb)
    for out1 in outNodes:
        ln_shape = (len(d_symb),) + d_affine[0].shape
        ln_coeff_lte = np.zeros(ln_shape).astype('float32')
        for out2 in outNodes:
            if(out2 != out1):
                ln_coeff_lte[:,out2,out1] = 1
                ln_coeff_lte[:,out2,out2] = -1
        d_ln_coeff_lte = cp.asarray(ln_coeff_lte)
        d_ln_coeff_gte = d_ln_coeff_lte.copy().astype('float32')
        layer = len(d_affine)
        while (layer != 1):  # layer zero is input and layer one is in already in terms of input
            # First relu of previous layer
            if (if_activation[layer - 1][1] == True):
                back_relu_GPU(d_symb[:,layer - 1], d_ln_coeff_lte, d_ln_coeff_gte)
            # Then affine of previous layer
            d_ineq_prev_gte = d_affine[layer - 1]
            d_ineq_prev_lte = d_affine[layer - 1]
            d_ln_coeff_lte, d_ln_coeff_gte = back_affine_GPU(d_ineq_prev_lte, d_ineq_prev_gte, d_ln_coeff_lte,
                                                            d_ln_coeff_gte,d_symb[:,layer - 1])
            layer -= 1
        d_lbs,d_ubs = get_bounds_GPU(d_ln_coeff_lte,d_ln_coeff_gte,d_l1_lb,d_l1_ub)
        lbs = cp.asnumpy(d_lbs)

        #print(f"DEBUG OUTCOME Node{out1} --> lbs:{d_lbs}; ubs:{d_ubs}")
        for init_id in range(len(d_symb)):
            if(outcomes[init_id]==None):
                flag = True
                for out2 in outNodes:
                    if (out2 != out1) and (lbs[init_id][out2]<=0.0):
                        flag = False
                        break;
                if(flag == True):
                    stmt = inv_var_index[(len(d_affine)-1,out1)]
                    outcomes[init_id] = stmt
    return outcomes

def active_convert(active_status,dims,inv_var_index):
    activated = []
    deactivated = []
    for init_id in range(len(active_status)):
        node_num = 3
        act = set()
        deact = set()
        for layer_index in range(1,len(dims[1:])):
            for neuron_index in range(1,dims[layer_index]):
                if(active_status[init_id,layer_index,neuron_index] == 0):
                    stmt = inv_var_index[(layer_index,neuron_index)]
                    val = Basic(node_num,[PyVar(stmt)])
                    deact.add(val)
                elif(active_status[init_id,layer_index,neuron_index] == 1):
                    stmt = inv_var_index[(layer_index,neuron_index)]
                    val = Basic(node_num, [PyVar(stmt)])
                    act.add(val)
                node_num+=1
            node_num+=1
        activated.append(act)
        deactivated.append(deact)
    return activated,deactivated


def getNetShape(nodes):
    NO_OF_LAYERS = 1
    MAX_NODES_IN_LAYER = 1
    CURR_NODED_IN_LAYER = 0
    flag = False
    for current in nodes:
        if isinstance(current, Function):
            for (node, eq) in zip(current.stmts[0], current.stmts[1]):
                if (flag):
                    flag = False
                    NO_OF_LAYERS += 1
                    MAX_NODES_IN_LAYER = max(MAX_NODES_IN_LAYER, CURR_NODED_IN_LAYER)
                    CURR_NODED_IN_LAYER = 0
                CURR_NODED_IN_LAYER += 1
        elif isinstance(current, Activation):
            flag = True
            continue
    MAX_NODES_IN_LAYER = max(MAX_NODES_IN_LAYER, CURR_NODED_IN_LAYER)
    return NO_OF_LAYERS,MAX_NODES_IN_LAYER

def fillInput(nodes,affine,dims,if_activation,var_index,MNIL):
    row_id,col_id,flag = (1,1,False)
    # The 4 in relu is for lessThan(slope,y-coeff);greaterThan(slope,y-coeff)
    # TO-DO: can make the assumption if one node in a layer has relu then all do
    for current in nodes:
        if isinstance(current, Function):
            for (node, eq) in zip(current.stmts[0], current.stmts[1]):
                if (flag):
                    flag = False
                    row_id += 1
                    col_id = 1
                coeffs = np.zeros((MNIL + 1,)).astype(np.float32)
                eq = texpr_to_dict(eq)
                for var, val in eq.items():
                    if var != '_':
                        r, c = var_index[str(var)]
                        if (r != row_id - 1):
                            raise NotImplementedError(
                                f"Affine should only be based on previous layer{row_id} But was on {r}.")
                        coeffs[c] += val
                    else:
                        coeffs[0] += val
                dims[row_id] += 1
                affine[row_id, col_id, :] = coeffs
                # print(f"Afiine->{str(node)}")
                var_index[str(node)] = (row_id, col_id)
                col_id += 1
                # TO-DO: do something more general than assume format X[N][N] for var name
        elif isinstance(current, Activation):
            flag = True
            r, c = var_index[str(current.stmts)]
            if_activation[r, c] = True
            # print(f"Relu->{str(current.stmts)}")
        else:
            # print(f"Others->{current.stmts}")
            '''What to do here
            for stmt in reversed(current.stmts):
            state = semantics.assume_call_semantics(stmt, state, manager)'''
            continue

def detailedPrintCondense( d_affine, d_symb, d_active_pattern, d_l1_lb, d_l1_ub, if_activation, symb,
                          var_index, inv_var_index, l1_lb, l1_ub):
    print(f"var_index = {var_index}")
    print(f"inv_var_index = {inv_var_index}")
    # Detailed print for DEBUG
    for i in range(1, len(d_affine)):
        print(f"\t\t LAYER {i} Input Equations")
        for j in range(1, len(d_affine[0])):
            print(
                f"Node: {j} -> if_activation: {if_activation[i, j]}\n eq: {ineq_str(d_affine[i, j], i, j, '=', i - 1, inv_var_index)} ")
        d_ineq_lte, d_ineq_gte = back_propagate_GPU(d_affine, d_symb, i, if_activation, d_active_pattern, d_l1_lb,
                                                    d_l1_ub)
        if (if_activation[i][1] == 1):
            d_lbs, d_ubs = get_bounds_GPU(d_ineq_lte, d_ineq_gte, d_l1_lb, d_l1_ub)
            relu_compute_GPU(d_lbs, d_ubs, d_symb[:, i], d_active_pattern[:, i], d_l1_lb, d_l1_ub)
            symb = cp.asnumpy(d_symb)

        ineq_lte = cp.asnumpy(d_ineq_lte)
        ineq_gte = cp.asnumpy(d_ineq_gte)
        init_id = 1

        print(f"\t\t LAYER {i} Substituted")
        for j in range(1, len(d_affine[0])):
            print(f"\tNode {j}")
            print(f" eq LTE L1: {ineq_str(ineq_lte[init_id][j], i, j, '>=', 0, inv_var_index)}")
            print(f" eq GTE L1: {ineq_str(ineq_gte[init_id][j], i, j, '<=', 0, inv_var_index)}")
            print(f" eq (LB,UB): {get_bounds_single(ineq_lte[init_id], ineq_gte[init_id], j, l1_lb[init_id], l1_ub[init_id])}")
        if (if_activation[i, 1] == 1):  # assuming if first node in a layer has activation then all do
            print(f"\t RELU-LAYER {i}")
            for j in range(1, len(d_affine[0])):
                print(f"\tNode {j}")
                print(f" SYMB  BOOL: {symb[init_id][i][j][0]}, LB: {symb[init_id][i][j][2]}, UB: {symb[init_id][i][j][1]}")
                if (symb[init_id][i][j][0] == 0.0):
                    print(f"Relu eq (LB,UB): {get_bounds_single(ineq_lte[init_id], ineq_gte[init_id], j, l1_lb[init_id], l1_ub[init_id])}")
                else:
                    print(f"Relu eq (LB,UB): ({symb[init_id][i][j][2]},{symb[init_id][i][j][1]})")
        # print stuff
        else:
            print(f"\t\t NO RELU ON LAYER {i}")
    #print(f"activation->{d_active_pattern}")

def miniPrintCondense(d_affine,d_symb,d_active_pattern,d_l1_lb,d_l1_ub,if_activation,l1_lb,l1_ub,symb):
    for i in range(1, len(d_affine)):
        d_ineq_lte, d_ineq_gte = back_propagate_GPU(d_affine, d_symb, i, if_activation, d_active_pattern, d_l1_lb,
                                                    d_l1_ub)
        if (if_activation[i][1] == 1):
            d_lbs, d_ubs = get_bounds_GPU(d_ineq_lte, d_ineq_gte, d_l1_lb, d_l1_ub)
            relu_compute_GPU(d_lbs, d_ubs, d_symb[:, i], d_active_pattern[:, i], d_l1_lb, d_l1_ub)
            symb = cp.asnumpy(d_symb)

        ineq_lte = cp.asnumpy(d_ineq_lte)
        ineq_gte = cp.asnumpy(d_ineq_gte)
        init_id = 1
        if (if_activation[i, 1] == 1):  # assuming if first node in a layer has activation then all do
            for j in range(1, len(d_affine[0])):
                if (symb[init_id][i][j][0] == 0.0):
                    print(f"Relu{i}:{j} eq (LB,UB): {get_bounds_single(ineq_lte[init_id], ineq_gte[init_id], j, l1_lb[init_id], l1_ub[init_id])}")
                else:
                    print(f"Relu{i}:{j} eq (LB,UB): ({symb[init_id][i][j][2]},{symb[init_id][i][j][1]})")
        else:
            for j in range(1, len(d_affine[0])):
                print(f"{i}:{j} eq (LB,UB): {get_bounds_single(ineq_lte[init_id], ineq_gte[init_id], j,l1_lb[init_id],l1_ub[init_id])}")

def noPrintCondense(d_affine, d_symb, i, if_activation, d_active_pattern,d_l1_lb,d_l1_ub):
    for i in range(1, len(d_affine)):
        d_ineq_lte, d_ineq_gte = back_propagate_GPU(d_affine, d_symb, i, if_activation, d_active_pattern,d_l1_lb,d_l1_ub)
        if (if_activation[i][1] == 1):
            d_lbs, d_ubs = get_bounds_GPU(d_ineq_lte, d_ineq_gte, d_l1_lb, d_l1_ub)
            relu_compute_GPU(d_lbs, d_ubs, d_symb[:,i], d_active_pattern[:,i], d_l1_lb, d_l1_ub)

def network_condense_GPU(nodes, initial,outputs):
    # equation[n1][n2] stores the bias and coeff of nodes of previous layer to form x[n1][n2] in order
    # if_activation[n1][n2] stores if there is an activation on x[n1][n2] (only relu considered for now)
    NO_OF_LAYERS, MAX_NODES_IN_LAYER = getNetShape(nodes)
    NO_OF_INITIALS = 7000
    print(f"NO_OF_LAYER:{NO_OF_LAYERS}; MNIL:{MAX_NODES_IN_LAYER}")
    var_index: dict(str, (int, int)) = dict()


    affine = np.zeros((NO_OF_LAYERS + 1, MAX_NODES_IN_LAYER + 1, MAX_NODES_IN_LAYER + 1)).astype(np.float32)
    symb = np.zeros((NO_OF_INITIALS,NO_OF_LAYERS + 1, MAX_NODES_IN_LAYER + 1, 3)).astype(np.float32)
    if_activation = np.zeros((NO_OF_LAYERS + 1, MAX_NODES_IN_LAYER + 1)).astype(np.float32)
    active_pattern = np.zeros((NO_OF_INITIALS,NO_OF_LAYERS + 1, MAX_NODES_IN_LAYER + 1)).astype(np.float32)
    l1_lb = np.zeros((NO_OF_INITIALS,MAX_NODES_IN_LAYER + 1)).astype(np.float32)
    l1_ub = np.zeros((NO_OF_INITIALS,MAX_NODES_IN_LAYER + 1)).astype(np.float32)
    dims = np.ones(NO_OF_LAYERS + 1).astype(np.int32)

    # obtain the lower bound and upper bound for input layer using "initial"
    # Assuming "initial" contains input from 0 to nth input in order.
    i = 1
    row_id, col_id, flag = (0, 1, False)
    for var, bound in initial.bounds.items():
        var_index[str(var)] = (row_id, col_id)
        col_id += 1

    for ini in range(NO_OF_INITIALS):
        i = 1
        for var, bound in initial.bounds.items():
            a = random.uniform(bound.lower, bound.upper)
            b = random.uniform(bound.lower, bound.upper)
            l1_lb[ini][i] = min(a, b)  # bound.lower
            l1_ub[ini][i] = max(a, b)  # bound.upper
            i += 1

    fillInput(nodes, affine, dims, if_activation, var_index, MAX_NODES_IN_LAYER)
    outNodes = set()
    for output in outputs:
        outNodes.add(var_index[str(output)][1])
    inv_var_index = {v: k for k, v in var_index.items()}

    d_affine = cp.asarray(affine)
    d_symb = cp.asarray(symb)
    d_active_pattern = cp.asarray(active_pattern)
    d_l1_lb = cp.asarray(l1_lb)
    d_l1_ub = cp.asarray(l1_ub)
    # Removes NumbaPerformanceWarning and others but slow down everything significantly.
    warnings.filterwarnings("ignore")
    #detailedPrintCondense(d_affine, d_symb, d_active_pattern, d_l1_lb, d_l1_ub, if_activation, symb, var_index,inv_var_index, l1_lb, l1_ub)
    #miniPrintCondense(d_affine, d_symb, d_active_pattern, d_l1_lb, d_l1_ub, if_activation, l1_lb, l1_ub, symb)
    noPrintCondense(d_affine, d_symb, i, if_activation, d_active_pattern, d_l1_lb, d_l1_ub)

    outcome = oneOutput(affine[-1],d_affine, d_symb, if_activation,d_l1_lb,d_l1_ub,outNodes,inv_var_index)
    #print(f"INSIDE activation -> {d_active_pattern}; dims -> {dims}")
    active_pattern = cp.asnumpy(d_active_pattern)
    activated, deactivated = active_convert(active_pattern,dims,inv_var_index)
    '''for i in range(NO_OF_INITIALS):
        print(f"l1_lb -> {d_l1_lb[i]}; l1_ub -> {d_l1_ub[i]}")
        print(f"activation->{active_pattern[i]}")
        print(f"GPU active:{activated[i]}; deactive:{deactivated[i]}; outcome:{outcome[i]}")
    # return activated, deactivated, outcome'''