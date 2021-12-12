import numpy as np
import cupy as cp
from numba import cuda
from libra.optimized.commons import texpr_to_dict, get_bounds_single, ineq_str,ineq_str_direct,get_bounds_single_neurify
from libra.core.cfg import Node, Function, Activation,Basic
from libra.core.expressions import VariableIdentifier
import warnings
from apronpy.var import PyVar
import copy
import random
from  libra.optimized import symbolic_gpu as smbG
from  libra.optimized import neurify_gpu as neuG
from  libra.optimized import deeppoly_gpu as dpG

def oneOutput(d_affine,d_relu_dp,d_relu_neu,d_symb,if_activation,d_l1_lb,d_l1_ub,outNodes,inv_var_index,domains):
    outcomes = [None] * len(d_l1_lb)
    for out1 in outNodes:
        ln_shape = (len(d_l1_lb),) + d_affine[0].shape
        ln_coeff_lte = np.zeros(ln_shape).astype('float32')
        for out2 in outNodes:
            if(out2 != out1):
                ln_coeff_lte[:,out2,out1] = 1
                ln_coeff_lte[:,out2,out2] = -1
        d_ln_coeff_lte = cp.asarray(ln_coeff_lte)
        d_ln_coeff_gte = d_ln_coeff_lte.copy().astype('float32')
        layer = len(d_affine)

        d_lbsL = cp.zeros((len(domains), len(d_l1_lb), len(d_affine[0])))
        d_ubsL = cp.zeros((len(domains), len(d_l1_lb), len(d_affine[0])))
        j = 0
        if ("DeepPoly" in domains):
            d_ineq_lte, d_ineq_gte = dpG.back_propagate_GPU1(d_ln_coeff_lte, d_ln_coeff_gte,d_affine, d_relu_dp, layer, if_activation)
            d_lbsL[j], d_ubsL[j] = dpG.get_bounds_GPU(d_ineq_lte, d_ineq_gte, d_l1_lb, d_l1_ub)
            j += 1
        if ("Symbolic" in domains):
            d_ineq_lte, d_ineq_gte = smbG.back_propagate_GPU1(d_ln_coeff_lte, d_ln_coeff_gte,d_affine, d_symb, layer, if_activation)
            d_lbsL[j], d_ubsL[j] = smbG.get_bounds_GPU(d_ineq_lte, d_ineq_gte, d_l1_lb, d_l1_ub)
            j += 1
        if ("Neurify" in domains):
            d_ineq_lte, d_ineq_gte = neuG.back_propagate_GPU1(d_ln_coeff_lte, d_ln_coeff_gte,d_affine, d_relu_neu, layer, if_activation)
            d_lbs_low, d_ubs_low = neuG.get_bounds_GPU(d_ineq_lte, d_ineq_lte, d_l1_lb, d_l1_ub)
            d_lbs_up, d_ubs_up = neuG.get_bounds_GPU(d_ineq_gte, d_ineq_gte, d_l1_lb, d_l1_ub)
            d_lbsL[j], d_ubsL[j] = d_lbs_low, d_ubs_up
            j += 1
        d_lbs, d_ubs = mergeBounds(d_lbsL, d_ubsL)
        lbs = cp.asnumpy(d_lbs)
        # print(f"DEBUG OUTCOME Node{out1} --> lbs:{d_lbs}; ubs:{d_ubs}")
        for init_id in range(len(d_l1_lb)):
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

def miniPrintCondense(d_affine, if_activation, d_l1_lb,d_l1_ub,domains,d_relu_dp,d_symb,d_relu_neu,d_active_pattern):
    d_lbsL = cp.zeros((len(domains), len(d_relu_dp), len(d_affine[0])))
    d_ubsL = cp.zeros((len(domains), len(d_relu_dp), len(d_affine[0])))
    for i in range(1, len(d_affine)):
        j = 0
        if ("DeepPoly" in domains):
            d_ineq_lte, d_ineq_gte = dpG.back_propagate_GPU(d_affine, d_relu_dp, i, if_activation)
            d_lbsL[j], d_ubsL[j] = dpG.get_bounds_GPU(d_ineq_lte, d_ineq_gte, d_l1_lb, d_l1_ub)
            j += 1
        if ("Symbolic" in domains):
            d_ineq_lte, d_ineq_gte = smbG.back_propagate_GPU(d_affine, d_symb, i, if_activation)
            d_lbsL[j], d_ubsL[j] = smbG.get_bounds_GPU(d_ineq_lte, d_ineq_gte, d_l1_lb, d_l1_ub)
            j += 1
        if ("Neurify" in domains):
            d_ineq_lte, d_ineq_gte = neuG.back_propagate_GPU(d_affine, d_relu_neu, i, if_activation)
            d_lbs_low, d_ubs_low = neuG.get_bounds_GPU(d_ineq_lte, d_ineq_lte, d_l1_lb, d_l1_ub)
            d_lbs_up, d_ubs_up = neuG.get_bounds_GPU(d_ineq_gte, d_ineq_gte, d_l1_lb, d_l1_ub)
            d_lbsL[j], d_ubsL[j] = d_lbs_low, d_ubs_up
            j += 1
        d_lbs, d_ubs = mergeBounds(d_lbsL, d_ubsL)
        if (if_activation[i][1] == 1):
            if ("DeepPoly" in domains):
                dpG.relu_compute_GPU(d_lbs, d_ubs, d_relu_dp[:, i], d_active_pattern[:, i], d_l1_lb, d_l1_ub)
            if ("Symbolic" in domains):
                smbG.relu_compute_GPU(d_lbs, d_ubs, d_symb[:, i], d_active_pattern[:, i], d_l1_lb, d_l1_ub)
            if ("Neurify" in domains):
                neuG.relu_compute_GPU(d_lbs, d_ubs_low, d_lbs_up, d_ubs, d_relu_neu[:, i], d_active_pattern[:, i, :],
                                      d_l1_lb, d_l1_ub)
        init_id = 1
        for j in range(1, len(d_affine[0])):
            print(f"Affine {i}:{j} eq (LB,UB): ({d_lbs[init_id][j]}, {d_ubs[init_id][j]})")

def mergeBounds(d_lbsL,d_ubsL):
    d_lbs = cp.amax(d_lbsL, axis=0)
    d_ubs = cp.amin(d_ubsL, axis=0)
    #print(f"DEBUG -> d_ubsL: {d_ubsL}\n d_lbsL: {d_lbsL};\n d_lbs: {d_lbs}\n d_ubs:{d_ubs}")
    return d_lbs,d_ubs

def noPrintCondense(d_affine, if_activation, d_l1_lb,d_l1_ub,domains,d_relu_dp,d_symb,d_relu_neu,d_active_pattern):
    d_lbsL = cp.zeros((len(domains), len(d_l1_lb),len(d_affine[0])))
    d_ubsL = cp.zeros((len(domains), len(d_l1_lb), len(d_affine[0])))
    for i in range(1, len(d_affine)):
        j = 0
        if ("DeepPoly" in domains):
            d_ineq_lte, d_ineq_gte = dpG.back_propagate_GPU(d_affine, d_relu_dp, i, if_activation)
            d_lbsL[j], d_ubsL[j] =   dpG.get_bounds_GPU(d_ineq_lte, d_ineq_gte, d_l1_lb, d_l1_ub)
            j += 1
        if ("Symbolic" in domains):
            d_ineq_lte, d_ineq_gte = smbG.back_propagate_GPU(d_affine, d_symb, i, if_activation)
            d_lbsL[j], d_ubsL[j] = smbG.get_bounds_GPU(d_ineq_lte, d_ineq_gte, d_l1_lb, d_l1_ub)
            j += 1
        if ("Neurify" in domains):
            d_ineq_lte, d_ineq_gte = neuG.back_propagate_GPU(d_affine, d_relu_neu, i,if_activation)
            d_lbs_low, d_ubs_low = neuG.get_bounds_GPU(d_ineq_lte, d_ineq_lte, d_l1_lb,d_l1_ub)
            d_lbs_up, d_ubs_up = neuG.get_bounds_GPU(d_ineq_gte, d_ineq_gte, d_l1_lb,d_l1_ub)
            d_lbsL[j], d_ubsL[j] = d_lbs_low,d_ubs_up
            j += 1
        d_lbs,d_ubs = mergeBounds(d_lbsL,d_ubsL)
        if (if_activation[i][1] == 1):
            if ("DeepPoly" in domains):
                dpG.relu_compute_GPU(d_lbs, d_ubs, d_relu_dp[:,i], d_active_pattern[:,i], d_l1_lb, d_l1_ub)
            if ("Symbolic" in domains):
                smbG.relu_compute_GPU(d_lbs, d_ubs, d_symb[:,i], d_active_pattern[:,i], d_l1_lb, d_l1_ub)
            if ("Neurify" in domains):
                neuG.relu_compute_GPU(d_lbs, d_ubs_low, d_lbs_up, d_ubs, d_relu_neu[:,i], d_active_pattern[:,i,:],d_l1_lb,d_l1_ub)
    #print(f"DP:\n{d_active_pattern_dp}")
    #print(f"SYM:\n{d_active_pattern_symb}")
    #print(f"NEU:\n{d_active_pattern_neu}")

def network_condense_GPU(nodes, initial,domains,outputs):
    # equation[n1][n2] stores the bias and coeff of nodes of previous layer to form x[n1][n2] in order
    # if_activation[n1][n2] stores if there is an activation on x[n1][n2] (only relu considered for now)
    NO_OF_LAYERS, MAX_NODES_IN_LAYER = getNetShape(nodes)
    NO_OF_INITIALS = 2**17
    print(f"NO_OF_LAYER:{NO_OF_LAYERS}; MAX_NODES_IN_LAYER:{MAX_NODES_IN_LAYER}; NO_OF_INITIALS:{NO_OF_INITIALS}")
    var_index: dict(str, (int, int)) = dict()

    affine = np.zeros((NO_OF_LAYERS + 1, MAX_NODES_IN_LAYER + 1, MAX_NODES_IN_LAYER + 1)).astype(np.float32)
    if_activation = np.zeros((NO_OF_LAYERS + 1, MAX_NODES_IN_LAYER + 1)).astype(np.float32)
    l1_lb = np.zeros((NO_OF_INITIALS,MAX_NODES_IN_LAYER + 1)).astype(np.float32)
    l1_ub = np.zeros((NO_OF_INITIALS,MAX_NODES_IN_LAYER + 1)).astype(np.float32)
    dims = np.ones(NO_OF_LAYERS + 1).astype(np.int32)
    active_pattern = np.zeros((NO_OF_INITIALS, NO_OF_LAYERS + 1, MAX_NODES_IN_LAYER + 1)).astype(np.float32)

    if ("DeepPoly" in domains):
        relu_dp = np.zeros((NO_OF_INITIALS,NO_OF_LAYERS + 1, MAX_NODES_IN_LAYER + 1, 4)).astype(np.float32)
    if ("Symbolic" in domains):
        symb = np.zeros((NO_OF_INITIALS,NO_OF_LAYERS + 1, MAX_NODES_IN_LAYER + 1, 3)).astype(np.float32)
    if ("Neurify" in domains):
        relu_neu = np.zeros((NO_OF_INITIALS,NO_OF_LAYERS + 1, MAX_NODES_IN_LAYER + 1, 4)).astype(np.float32)

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

    # can remove layer 0 as it has no inequations
    # All these print are for debug mode. Actual will only activation pattern.
    d_symb,d_relu_dp,d_relu_neu= (None,None,None)
    d_affine = cp.asarray(affine)
    d_active_pattern = cp.asarray(active_pattern)
    d_l1_lb = cp.asarray(l1_lb)
    d_l1_ub = cp.asarray(l1_ub)
    if ("DeepPoly" in domains):
        d_relu_dp = cp.asarray(relu_dp)
    if ("Symbolic" in domains):
        d_symb = cp.asarray(symb)
    if ("Neurify" in domains):
        d_relu_neu = cp.asarray(relu_neu)

    # Removes NumbaPerformanceWarning and others but slow down everything significantly.
    warnings.filterwarnings("ignore")
    #miniPrintCondense(d_affine, if_activation, d_l1_lb,d_l1_ub,domains,d_relu_dp,d_symb,d_relu_neu,d_active_pattern)
    #noPrintCondense(d_affine, if_activation, d_l1_lb,d_l1_ub,domains,d_relu_dp,d_symb,d_relu_neu,d_active_pattern)

    #print(f"activation->{d_active_pattern}")
    outcome = oneOutput(d_affine,d_relu_dp,d_relu_neu,d_symb,if_activation,d_l1_lb,d_l1_ub,outNodes,inv_var_index,domains)
    active_pattern = cp.asnumpy(d_active_pattern)
    activated, deactivated = dpG.active_convert(active_pattern, dims, inv_var_index)
    '''for i in range(NO_OF_INITIALS):
        print(f"l1_lb -> {d_l1_lb[i]}; l1_ub -> {d_l1_ub[i]}")
        print(f"activation->{active_pattern[i]}")
        print(f"GPU active:{activated[i]}; deactive:{deactivated[i]}; outcome:{outcome[i]}")
    # return activated, deactivated, outcome'''



