import numpy as np
import cupy as cp
from numba import cuda
from libra.optimized.commons import texpr_to_dict, get_bounds_single, ineq_str
from libra.core.cfg import Node, Function, Activation,Basic
from libra.core.expressions import VariableIdentifier
import warnings
from apronpy.var import PyVar
import copy


def get_bounds_GPU(d_l1_lte, d_l1_gte, d_l1_lb, d_l1_ub):
    @cuda.jit
    def bound_helper(l1_lte, l1_gte, l1_lb, l1_ub, lbs, ubs):
        id = cuda.grid(1)
        if (id >= len(lbs)):
            return
        lbs[id] = l1_lte[id, 0]
        for i in range(1,len(l1_lte[id])):
            if(l1_lte[id,i]<0):
                lbs[id] += l1_lte[id,i] * l1_ub[i]
            else:
                lbs[id] += l1_lte[id,i] * l1_lb[i]
        ubs[id] = l1_gte[id, 0]
        for i in range(1,len(l1_gte[id])):
            if(l1_gte[id,i]>0):
                ubs[id] += l1_gte[id,i] * l1_ub[i]
            else:
                ubs[id] += l1_gte[id,i] * l1_lb[i]

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


def relu_compute_GPU(d_l1_lte, d_l1_gte, d_symb_layer, d_active_pattern,d_l1_lb,d_l1_ub):
    @cuda.jit
    def relu_compute_helper(lbs, ubs, symb_layer, active_pattern):
        id = cuda.grid(1)
        if (id< 1 or id >= len(ubs)):
            return
        if (ubs[id] < 0):
            symb_layer[id] = (0.0, 0.0, 0.0)
            active_pattern[id] = 0
            return  # as initialized with zeros
        if (lbs[id] > 0):
            symb_layer[id] = (0.0, 0.0, 0.0)
            active_pattern[id] = 1
            return
        active_pattern[id] = 2
        symb_layer[id] = (1.0, ubs[id], 0.0)

    d_lbs, d_ubs = get_bounds_GPU(d_l1_lte, d_l1_gte,d_l1_lb,d_l1_ub)
    tpb = (min(1024, len(d_l1_lte)),)
    bpg = (int(np.ceil(len(d_l1_lte) / tpb[0])),)
    relu_compute_helper[bpg, tpb](d_lbs,
                                  d_ubs,
                                  d_symb_layer,
                                  d_active_pattern)
    return d_symb_layer, d_active_pattern

@cuda.jit
def back_affine_helper(i, l1_lte, l1_gte, ln_coeff_lte, ln_coeff_gte, ineq_prev_lte, ineq_prev_gte,symb_layer):
    k, p = cuda.grid(2)
    i = i[0]
    if (k >= len(l1_lte) or p >= len(ineq_prev_lte[i])):
        return
    if (symb_layer[i][0] == 0.0):
        if (ln_coeff_lte[k][i] > 0):
            l1_lte[k][p] += ln_coeff_lte[k][i] * ineq_prev_lte[i][p]
        else:
            l1_lte[k][p] += ln_coeff_lte[k][i] * ineq_prev_gte[i][p]
        if (ln_coeff_gte[k][i] > 0):
            l1_gte[k][p] += ln_coeff_gte[k][i] * ineq_prev_gte[i][p]
        else:
            l1_gte[k][p] += ln_coeff_gte[k][i] * ineq_prev_lte[i][p]

def back_affine_GPU(d_ineq_prev_lte, d_ineq_prev_gte, d_ln_coeff_lte, d_ln_coeff_gte,d_symb_layer):
    d_l1_lte = cp.zeros(d_ineq_prev_gte.shape)
    d_l1_gte = cp.zeros(d_ineq_prev_gte.shape)
    d_l1_lte[:, 0] = d_ln_coeff_lte[:, 0]
    d_l1_gte[:, 0] = d_ln_coeff_gte[:, 0]

    cuda_iters = (len(d_l1_lte), len(d_ineq_prev_lte[1]))
    tpb = (min(32, cuda_iters[0]), min(32, cuda_iters[1]))
    bpg = (int(np.ceil(cuda_iters[0] / tpb[0])), int(np.ceil(cuda_iters[1] / tpb[1])))

    for i in range(1, len(d_l1_lte)):
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
    i = cuda.grid(1)
    if (i < 1 or i >= len(ln_coeff_lte)):
        return
    for j in range(1, len(symb_layer)):
        if(symb_layer[j][0] == 1.0):
            if (ln_coeff_lte[i][j] > 0):
                ln_coeff_lte[i][0] += 0 * ln_coeff_lte[i][j]
            else:
                ln_coeff_lte[i][0] += symb_layer[j][1] * ln_coeff_lte[i][j]
            if (ln_coeff_gte[i][j] > 0):
                ln_coeff_gte[i][0] += symb_layer[j][1] * ln_coeff_gte[i][j]
            else:
                ln_coeff_gte[i][0] += 0 * ln_coeff_gte[i][j]

def back_relu_GPU(d_symb_layer, d_ln_coeff_lte, d_ln_coeff_gte):
    cuda_iters1 = (len(d_ln_coeff_lte),)
    tpb1 = (min(1024, cuda_iters1[0]),)
    bpg1 = (int(np.ceil(cuda_iters1[0] / tpb1[0])),)
    back_relu_helper[bpg1, tpb1](d_symb_layer,
                                      d_ln_coeff_lte,
                                      d_ln_coeff_gte)



def back_propagate_GPU(d_affine, d_symb, layer: int, if_activation, d_active_pattern,d_l1_lb,d_l1_ub):
    # shift the CP creation to caller.
    d_ln_coeff_lte = d_affine[layer].copy().astype('float32')  # Need to create copies
    d_ln_coeff_gte = d_affine[layer].copy().astype('float32')
    layer_t = layer
    while (layer != 1):  # layer zero is input and layer one is in already in terms of input
        # First relu of previous layer
        if (if_activation[layer - 1][1] == True):
            back_relu_GPU(d_symb[layer - 1], d_ln_coeff_lte, d_ln_coeff_gte)
        # Then affine of previous layer
        d_ineq_prev_gte = d_affine[layer - 1]
        d_ineq_prev_lte = d_affine[layer - 1]
        d_ln_coeff_lte, d_ln_coeff_gte = back_affine_GPU(d_ineq_prev_lte, d_ineq_prev_gte, d_ln_coeff_lte,
                                                         d_ln_coeff_gte,d_symb[layer - 1])
        layer -= 1
    if (if_activation[layer_t][1] == 1):
        relu_compute_GPU(d_ln_coeff_lte, d_ln_coeff_gte, d_symb[layer_t], d_active_pattern[layer_t],d_l1_lb,d_l1_ub)
    else:
        pass
    #return d_active_pattern
    ''''# Different return for debug purposes'''
    ln_coeff_gte = cp.asnumpy(d_ln_coeff_gte).astype(np.float32)
    ln_coeff_lte = cp.asnumpy(d_ln_coeff_lte).astype(np.float32)
    return ln_coeff_lte, ln_coeff_gte

#Simplification Assumption: output layer has 2 nodes.
def oneOutput(last,d_affine,d_relu,if_activation,d_l1_lb,d_l1_ub):
    ln_coeff_lte = np.zeros(d_affine[1].shape).astype('float32')
    ln_coeff_lte[1] = np.subtract(last[1] , last[2])
    ln_coeff_lte[2] = np.subtract(last[2] , last[1])
    d_ln_coeff_lte = cp.asarray(ln_coeff_lte)
    d_ln_coeff_gte = d_ln_coeff_lte.copy().astype('float32')
    layer = len(d_affine)-1
    layer_t = layer
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


    d_lbs,d_ubs=get_bounds_GPU(d_ln_coeff_lte,d_ln_coeff_gte,d_l1_lb,d_l1_ub)
    lbs = cp.asnumpy(d_lbs)
    ubs = cp.asnumpy(d_ubs)
    ln_coeff_gte = cp.asnumpy(d_ln_coeff_gte).astype(np.float32)
    ln_coeff_lte = cp.asnumpy(d_ln_coeff_lte).astype(np.float32)
    '''print(f"DEBUG --> l1_lb: {d_l1_lb}; l1_ub:{d_l1_ub}")
    print(f" eq LTE L1: {ineq_str(ln_coeff_lte[1], 4, 1, '>=', 0)}")
    print(f" eq GTE L1: {ineq_str(ln_coeff_gte[1], 4, 1, '<=', 0)}")
    print(f"DEBUG --> lbs:{d_lbs}; ubs:{d_ubs}")'''
    if(lbs[1]>0.0):
        stmt = "x" + str(len(d_affine)-1) + str(1)
        print(f"OneOutput -> {stmt}")
        return VariableIdentifier(stmt)
    elif(lbs[2]>0.0):
        stmt = "x" + str(len(d_affine) - 1) + str(2)
        print(f"OneOutput -> {stmt}")
        return VariableIdentifier(stmt)
    else:
        print(f"BOTH ELEMENT")
        return  None



def active_convert(active_status,dims):
    activated = set()
    deactivated = set()
    node_num = 3
    for layer_index in range(1,len(dims[1:])):
        for neuron_index in range(1,dims[layer_index]):
            if(active_status[layer_index,neuron_index] == 0):
                stmt = "x"+str(layer_index)+str(neuron_index)
                val = Basic(node_num,[PyVar(stmt)])
                deactivated.add(val)
            elif(active_status[layer_index,neuron_index] == 1):
                stmt = "x" + str(layer_index) + str(neuron_index)
                val = Basic(node_num, [PyVar(stmt)])
                activated.add(val)
            node_num+=1
        node_num+=1
    return activated,deactivated

def network_condense_GPU(nodes, initial):
    # equation[n1][n2] stores the bias and coeff of nodes of previous layer to form x[n1][n2] in order
    # if_activation[n1][n2] stores if there is an activation on x[n1][n2] (only relu considered for now)

    NO_OF_LAYERS = 1
    MAX_NODES_IN_LAYER = 1
    CURR_NODED_IN_LAYER = 0
    flag = False
    for current in nodes:
        if isinstance(current, Function):
            for (node, eq) in zip(current.stmts[0], current.stmts[1]):
                if(flag):
                    flag = False
                    NO_OF_LAYERS += 1
                    MAX_NODES_IN_LAYER = max(MAX_NODES_IN_LAYER, CURR_NODED_IN_LAYER)
                    CURR_NODED_IN_LAYER = 0
                CURR_NODED_IN_LAYER += 1
        elif isinstance(current, Activation):
            flag = True
            continue
    print(f"NO_OF_LAYER:{NO_OF_LAYERS}; MNIL:{MAX_NODES_IN_LAYER}")
    var_index: dict(str,(int,int)) = dict()
    row_id = 0
    col_id = 1
    flag = False


    affine = np.zeros((NO_OF_LAYERS + 1, MAX_NODES_IN_LAYER + 1, MAX_NODES_IN_LAYER + 1)).astype(np.float32)
    symb = np.zeros((NO_OF_LAYERS + 1, MAX_NODES_IN_LAYER + 1,3)).astype(np.float32)
    if_activation = np.zeros((NO_OF_LAYERS + 1, MAX_NODES_IN_LAYER + 1)).astype(np.float32)
    active_pattern = np.zeros((NO_OF_LAYERS + 1, MAX_NODES_IN_LAYER + 1)).astype(np.float32)
    l1_lb = np.zeros(MAX_NODES_IN_LAYER+1).astype(np.float32)
    l1_ub = np.zeros(MAX_NODES_IN_LAYER+1).astype(np.float32)
    dims = np.ones(NO_OF_LAYERS+1).astype(np.int32)

    # obtain the lower bound and upper bound for input layer using "initial"
    #Assuming "initial" contains input from 0 to nth input in order.
    i = 1
    for var,bound in initial.bounds.items():
        l1_lb[i] = bound.lower
        l1_ub[i] = bound.upper
        var_index[str(var)] = (row_id, col_id)
        col_id += 1
        i+=1
    row_id = 1
    col_id = 1
    # The 4 in relu is for lessThan(slope,y-coeff);greaterThan(slope,y-coeff)
    # TO-DO: can make the assumption if one node in a layer has relu then all do
    for current in nodes:
        if isinstance(current, Function):
            for (node, eq) in zip(current.stmts[0], current.stmts[1]):
                if(flag):
                    flag = False
                    row_id += 1
                    col_id = 1
                coeffs = np.zeros((MAX_NODES_IN_LAYER + 1,)).astype(np.float32)
                eq = texpr_to_dict(eq)
                for var, val in eq.items():
                    if var != '_':
                        r,c = var_index[str(var)]
                        if(r != row_id-1):
                            raise NotImplementedError(f"Affine should only be based on previous layer{row_id} But was on {r}.")
                        coeffs[c] += val
                    else:
                        coeffs[0] += val

                dims[row_id] += 1
                affine[row_id, col_id,:] = coeffs
                #print(f"Afiine->{str(node)}")
                var_index[str(node)] = (row_id,col_id)
                col_id += 1
                # TO-DO: do something more general than assume format X[N][N] for var name
        elif isinstance(current, Activation):
            flag = True
            r,c = var_index[str(current.stmts)]
            if_activation[r,c] = True
            #print(f"Relu->{str(current.stmts)}")
        else:
            #print(f"Others->{current.stmts}")
            '''What to do here
			for stmt in reversed(current.stmts):
			state = semantics.assume_call_semantics(stmt, state, manager)'''
            continue
    print(f"var_index = {var_index}")
    inv_var_index = {v: k for k, v in var_index.items()}
    print(f"inv_var_index = {inv_var_index}")
    # can remove layer 0 as it has no inequations
    # All these print are for debug mode. Actual will only activation pattern.
    d_affine = cp.asarray(affine)
    d_symb = cp.asarray(symb)
    d_active_pattern = cp.asarray(active_pattern)

    d_l1_lb = cp.asarray(l1_lb)
    d_l1_ub = cp.asarray(l1_ub)

    #Detailed print for DEBUG
    for i in range(1, len(affine)):

        print(f"\t\t LAYER {i} Input Equations")
        for j in range(1, len(affine[0])):
            print(
                f"Node: {j} -> if_activation: {if_activation[i, j]}\n eq: {ineq_str(d_affine[i, j], i, j, '=', i - 1,inv_var_index)} ")
        ineq_lte, ineq_gte = back_propagate_GPU(d_affine, d_symb, i, if_activation, d_active_pattern,d_l1_lb,d_l1_ub)
        symb[i] = cp.asnumpy(d_symb[i])
        print(f"\t\t LAYER {i} Substituted")
        for j in range(1, len(affine[0])):
            print(f"\tNode {j}")
            print(f" eq LTE L1: {ineq_str(ineq_lte[j], i, j, '>=', 0,inv_var_index)}")
            print(f" eq GTE L1: {ineq_str(ineq_gte[j], i, j, '<=', 0,inv_var_index)}")
            print(
                f" eq (LB,UB): {get_bounds_single(ineq_lte, ineq_gte, j)}")  # Performing the whole debug-print segment in CPU will be removed later.

        if (if_activation[i, 1] == 1):  # assuming if first node in a layer has activation then all do
            print(f"\t RELU-LAYER {i}")
            for j in range(1, len(affine[0])):
                print(f"\tNode {j}")
                print(f" SYMB  BOOL: {symb[i][j][0]}, LB: {symb[i][j][2]}, UB: {symb[i][j][1]}")
                if(symb[i][j][0] == 0.0):
                    print(f"Relu eq (LB,UB): {get_bounds_single(ineq_lte, ineq_gte, j)}")
                else:
                    print(f"Relu eq (LB,UB): ({symb[i][j][2]},{symb[i][j][1]})")

        # print stuff
        else:
            print(f"\t\t NO RELU ON LAYER {i}")
    print(f"activation->{d_active_pattern}")
    '''
    warnings.filterwarnings("ignore")                       #Removes NumbaPerformanceWarning and others but slow down everything significantly.
    for i in range(1, len(affine)):
        back_propagate_GPU(d_affine, d_relu, i, if_activation, d_active_pattern,d_l1_lb,d_l1_ub)
    #a = np.subtract(affine[-1,1],affine[-1,2])
    outcome = oneOutput(affine[-1],d_affine, d_relu, if_activation,d_l1_lb,d_l1_ub)
    #print(f"INSIDE activation -> {d_active_pattern}; dims -> {dims}")
    active_pattern = cp.asnumpy(d_active_pattern)
    activated, deactivated = active_convert(active_pattern,dims)
    #print(f"INSIDE activated = {activated}, deactivated = {deactivated}")
    return activated,deactivated,outcome'''
