import numpy as np
import cupy as cp
from numba import cuda
from libra.optimized.commons import texpr_to_dict, get_bounds_single, ineq_str,ineq_str_direct,get_bounds_single_neurify
from libra.core.cfg import Node, Function, Activation,Basic
from libra.core.expressions import VariableIdentifier
import warnings
from apronpy.var import PyVar
import copy
from  libra.optimized import symbolic_gpu as smbG
from  libra.optimized import neurify_gpu as neuG
from  libra.optimized import deeppoly_gpu as dpG

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

def miniPrintCondense(d_affine, if_activation,d_active_pattern, d_l1_lb,d_l1_ub,domains,d_relu_dp = None,d_symb = None,d_relu_neu=None, d_l1_lb_neu=None,d_l1_ub_neu=None):
    dpG = DeepPolyGPU()
    smbG = SymbolicGPU()
    neuG = NeurifyGPU()
    d_lbsL = cp.zeros((len(domains), len(d_affine[0])))
    d_ubsL = cp.zeros((len(domains), len(d_affine[0])))
    for i in range(1, len(d_affine)):
        j = 0
        if ("DeepPoly" in domains):
            d_lbsL[j], d_ubsL[j], ineq_lte, ineq_gte = dpG.back_propagate_GPU(d_affine, d_relu_dp, i, if_activation,
                                                                              d_active_pattern, d_l1_lb, d_l1_ub)
            j += 1
        if ("Symbolic" in domains):
            d_lbsL[j], d_ubsL[j], ineq_lte, ineq_gte = smbG.back_propagate_GPU(d_affine, d_symb, i, if_activation,
                                                                               d_active_pattern, d_l1_lb, d_l1_ub)
            j += 1
        if ("Neurify" in domains):
            d_lbs_low, d_ubs_low, d_lbs_up, d_ubs_up, ineq_lte, ineq_gte = neuG.back_propagate_GPU(d_affine,
                                                                                                   d_relu_neu, i,
                                                                                                   if_activation,
                                                                                                   d_active_pattern,
                                                                                                   d_l1_lb_neu,
                                                                                                   d_l1_ub_neu)
            d_lbsL[j], d_ubsL[j] = d_lbs_low, d_ubs_up
            j += 1
        d_lbs, d_ubs = mergeBounds(d_lbsL, d_ubsL)
        if ("DeepPoly" in domains):
            dpG.relu_compute_GPU(d_lbs, d_ubs, d_relu_dp[i], d_active_pattern, d_l1_lb, d_l1_ub)
        if ("Symbolic" in domains):
            smbG.relu_compute_GPU(d_lbs, d_ubs, d_symb[i], d_active_pattern, d_l1_lb, d_l1_ub)
        if ("Neurify" in domains):
            neuG.relu_compute_GPU(d_lbs, d_ubs_low, d_lbs_up, d_ubs, d_relu_neu[i], d_active_pattern, d_l1_lb_neu,
                                  d_l1_ub_neu)

            for j in range(1, len(d_affine[0])):
                print(f"Affine {i}:{j} eq (LB,UB): ({d_lbs[j]}, {d_ubs[j]})")

def mergeBounds(d_lbsL,d_ubsL):
    d_lbs = cp.amax(d_lbsL, axis=0)
    d_ubs = cp.amin(d_ubsL, axis=0)
    #print(f"DEBUG -> d_ubsL: {d_ubsL}\n d_lbsL: {d_lbsL};\n d_lbs: {d_lbs}\n d_ubs:{d_ubs}")
    return d_lbs,d_ubs

def noPrintCondense(d_affine, if_activation, d_l1_lb,d_l1_ub,domains,d_relu_dp,d_symb,d_relu_neu, d_l1_lb_neu,d_l1_ub_neu,d_active_pattern_dp,d_active_pattern_symb,d_active_pattern_neu):

    d_lbsL = cp.zeros((len(domains), len(d_affine[0])))
    d_ubsL = cp.zeros((len(domains), len(d_affine[0])))
    for i in range(1, len(d_affine)):
        j = 0
        if ("DeepPoly" in domains):
            d_ineq_lte, d_ineq_gte = dpG.back_propagate_GPU(d_affine, d_relu_dp, i, if_activation,d_active_pattern_dp, d_l1_lb,d_l1_ub)
            d_lbsL[j], d_ubsL[j] =   dpG.get_bounds_GPU(d_ineq_lte, d_ineq_gte, d_l1_lb, d_l1_ub)
            j += 1
        if ("Symbolic" in domains):
            d_ineq_lte, d_ineq_gte = smbG.back_propagate_GPU(d_affine, d_symb, i, if_activation,d_active_pattern_symb, d_l1_lb, d_l1_ub)
            d_lbsL[j], d_ubsL[j] = smbG.get_bounds_GPU(d_ineq_lte, d_ineq_gte, d_l1_lb, d_l1_ub)
            j += 1
        if ("Neurify" in domains):
            d_ineq_lte, d_ineq_gte = neuG.back_propagate_GPU(d_affine, d_relu_neu, i,if_activation,d_active_pattern_neu,d_l1_lb_neu,d_l1_ub_neu)
            d_lbs_low, d_ubs_low = neuG.get_bounds_GPU(d_ineq_lte, d_ineq_lte, d_l1_lb, d_l1_ub)
            d_lbs_up, d_ubs_up = neuG.get_bounds_GPU(d_ineq_gte, d_ineq_gte, d_l1_lb, d_l1_ub)
            d_lbsL[j], d_ubsL[j] = d_lbs_low,d_ubs_up
            j += 1
        d_lbs,d_ubs = mergeBounds(d_lbsL,d_ubsL)
        if (if_activation[i][1] == 1):
            if ("DeepPoly" in domains):
                dpG.relu_compute_GPU(d_lbs, d_ubs, d_relu_dp[i], d_active_pattern_dp[i], d_l1_lb, d_l1_ub)
            if ("Symbolic" in domains):
                smbG.relu_compute_GPU(d_lbs, d_ubs, d_symb[i], d_active_pattern_symb[i], d_l1_lb, d_l1_ub)
            if ("Neurify" in domains):
                neuG.relu_compute_GPU(d_lbs, d_ubs_low, d_lbs_up, d_ubs, d_relu_neu[i], d_active_pattern_neu[i], d_l1_lb_neu,d_l1_ub_neu)
    print(f"DP:\n{d_active_pattern_dp}")
    print(f"SYM:\n{d_active_pattern_symb}")
    print(f"NEU:\n{d_active_pattern_neu}")

def network_condense_GPU(nodes, initial,domains,outputs):
    # equation[n1][n2] stores the bias and coeff of nodes of previous layer to form x[n1][n2] in order
    # if_activation[n1][n2] stores if there is an activation on x[n1][n2] (only relu considered for now)
    NO_OF_LAYERS, MAX_NODES_IN_LAYER = getNetShape(nodes)
    print(f"NO_OF_LAYER:{NO_OF_LAYERS}; MNIL:{MAX_NODES_IN_LAYER}")
    var_index: dict(str, (int, int)) = dict()
    row_id,col_id,flag = (0,1,False)

    affine = np.zeros((NO_OF_LAYERS + 1, MAX_NODES_IN_LAYER + 1, MAX_NODES_IN_LAYER + 1)).astype(np.float32)
    if_activation = np.zeros((NO_OF_LAYERS + 1, MAX_NODES_IN_LAYER + 1)).astype(np.float32)
    l1_lb = np.zeros(MAX_NODES_IN_LAYER + 1).astype(np.float32)
    l1_ub = np.zeros(MAX_NODES_IN_LAYER + 1).astype(np.float32)
    dims = np.ones(NO_OF_LAYERS + 1).astype(np.int32)

    if ("DeepPoly" in domains):
        relu_dp = np.zeros((NO_OF_LAYERS + 1, MAX_NODES_IN_LAYER + 1, 4)).astype(np.float32)
        active_pattern_dp = np.zeros((NO_OF_LAYERS + 1, MAX_NODES_IN_LAYER + 1)).astype(np.float32)

    if ("Symbolic" in domains):
        symb = np.zeros((NO_OF_LAYERS + 1, MAX_NODES_IN_LAYER + 1, 3)).astype(np.float32)
        active_pattern_symb = np.zeros((NO_OF_LAYERS + 1, MAX_NODES_IN_LAYER + 1)).astype(np.float32)

    if ("Neurify" in domains):
        relu_neu = np.zeros((NO_OF_LAYERS + 1, MAX_NODES_IN_LAYER + 1, 4)).astype(np.float32)
        l1_lb_neu = np.zeros((MAX_NODES_IN_LAYER + 1, 2)).astype(np.float32)  # bounds for LOW
        l1_ub_neu = np.zeros((MAX_NODES_IN_LAYER + 1, 2)).astype(np.float32)  # bounds for UP
        active_pattern_neu = np.zeros((2, NO_OF_LAYERS + 1, MAX_NODES_IN_LAYER + 1)).astype(np.float32)

    # obtain the lower bound and upper bound for input layer using "initial"
    # Assuming "initial" contains input from 0 to nth input in order.
    i = 1
    for var, bound in initial.bounds.items():
        l1_lb[i] = bound.lower
        l1_ub[i] = bound.upper
        if ("Neurify" in domains):
            l1_lb_neu[i][0] = bound.lower
            l1_lb_neu[i][1] = bound.lower
            l1_ub_neu[i][0] = bound.upper
            l1_ub_neu[i][1] = bound.upper
        var_index[str(var)] = (row_id, col_id)
        col_id += 1
        i += 1
    fillInput(nodes, affine, dims, if_activation, var_index, MAX_NODES_IN_LAYER)
    outNodes = set()
    for output in outputs:
        outNodes.add(var_index[str(output)][1])
    inv_var_index = {v: k for k, v in var_index.items()}

    # can remove layer 0 as it has no inequations
    # All these print are for debug mode. Actual will only activation pattern.
    d_symb,d_relu_dp,d_relu_neu,d_l1_ub_neu,d_l1_lb_neu,d_active_pattern_dp,d_active_pattern_symb,d_active_pattern_neu = (None,None,None,None,None,None,None,None)
    d_affine = cp.asarray(affine)

    d_l1_lb = cp.asarray(l1_lb)
    d_l1_ub = cp.asarray(l1_ub)
    if ("DeepPoly" in domains):
        d_relu_dp = cp.asarray(relu_dp)
        d_active_pattern_dp = cp.asarray(active_pattern_dp)
    if ("Symbolic" in domains):
        d_symb = cp.asarray(symb)
        d_active_pattern_symb = cp.asarray(active_pattern_symb)
    if ("Neurify" in domains):
        d_relu_neu = cp.asarray(relu_neu)
        d_l1_ub_neu = cp.asarray(l1_ub_neu)
        d_l1_lb_neu = cp.asarray(l1_lb_neu)
        d_active_pattern_neu = cp.asarray(active_pattern_neu)

    # Removes NumbaPerformanceWarning and others but slow down everything significantly.
    warnings.filterwarnings("ignore")
    #miniPrintCondense(d_affine, if_activation,d_active_pattern, d_l1_lb,d_l1_ub,domains,d_relu_dp = d_relu_dp,d_symb = d_symb,d_relu_neu=d_relu_neu, d_l1_lb_neu=d_l1_lb_neu,d_l1_ub_neu=d_l1_ub_neu)
    noPrintCondense(d_affine, if_activation, d_l1_lb,d_l1_ub,domains,d_relu_dp,d_symb,d_relu_neu,d_l1_lb_neu,d_l1_ub_neu,d_active_pattern_dp,d_active_pattern_symb,d_active_pattern_neu)

    #print(f"activation->{d_active_pattern}")
    #outcome = oneOutput(affine[-1], d_affine, d_relu, if_activation, d_l1_lb, d_l1_ub)
    active_pattern_dp = cp.asnumpy(d_active_pattern_dp)
    activated, deactivated = dpG.active_convert(active_pattern_dp, dims, inv_var_index)
    # print(f"GPU active:{activated}; deactive:{deactivated}; outcome:{outcome}")
    return activated, deactivated, None


