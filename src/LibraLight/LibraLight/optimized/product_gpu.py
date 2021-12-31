import numpy as np
import cupy as cp
from numba import cuda
import warnings
from optimized import deeppoly_gpu as dpG
from optimized import symbolic_gpu as symG
from optimized import neurify_gpu as neuG
from optimized import commons
from colorama import Style, Fore

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
            d_ineq_lte, d_ineq_gte = symG.back_propagate_GPU1(d_ln_coeff_lte, d_ln_coeff_gte,d_affine, d_symb, layer, if_activation)
            d_lbsL[j], d_ubsL[j] = symG.get_bounds_GPU(d_ineq_lte, d_ineq_gte, d_l1_lb, d_l1_ub)
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

def miniPrintCondense(d_affine, if_activation,d_if_activation, d_l1_lb,d_l1_ub,domains,d_relu_dp,d_symb,d_relu_neu,d_active_pattern,NO_OF_INITIALS):
    d_lbsL = cp.zeros((len(domains), NO_OF_INITIALS, len(d_affine[0])))
    d_ubsL = cp.zeros((len(domains), NO_OF_INITIALS, len(d_affine[0])))
    init_id = 0
    print(f"init_id-> {init_id}; lbs -> {d_l1_lb[init_id]}; ubs -> {d_l1_ub[init_id]}")
    for i in range(1, len(d_affine)):
        j = 0
        if ("DeepPoly" in domains):
            d_ineq_lte, d_ineq_gte = dpG.back_propagate_GPU(d_affine, d_relu_dp, i, if_activation)
            d_lbsL[j], d_ubsL[j] = dpG.get_bounds_GPU(d_ineq_lte, d_ineq_gte, d_l1_lb, d_l1_ub)
            j += 1
        if ("Symbolic" in domains):
            d_ineq_lte, d_ineq_gte = symG.back_propagate_GPU(d_affine, d_symb, i, if_activation)
            d_lbsL[j], d_ubsL[j] = symG.get_bounds_GPU(d_ineq_lte, d_ineq_gte, d_l1_lb, d_l1_ub)
            j += 1
        if ("Neurify" in domains):
            d_ineq_lte, d_ineq_gte = neuG.back_propagate_GPU(d_affine, d_relu_neu, i, if_activation)
            d_lbs_low, d_ubs_low = neuG.get_bounds_GPU(d_ineq_lte, d_ineq_lte, d_l1_lb, d_l1_ub)
            d_lbs_up, d_ubs_up = neuG.get_bounds_GPU(d_ineq_gte, d_ineq_gte, d_l1_lb, d_l1_ub)
            d_lbsL[j], d_ubsL[j] = d_lbs_low, d_ubs_up
            j += 1
        d_lbs, d_ubs = mergeBounds(d_lbsL, d_ubsL)
        if (any(if_activation[i])):
            if ("DeepPoly" in domains):
                dpG.relu_compute_GPU(d_lbs, d_ubs, d_relu_dp[:, i], d_active_pattern[:, i],d_if_activation[i])
            if ("Symbolic" in domains):
                symG.relu_compute_GPU(d_lbs, d_ubs, d_symb[:, i], d_active_pattern[:, i], d_l1_lb, d_l1_ub,d_if_activation[i])
            if ("Neurify" in domains):
                neuG.relu_compute_GPU(d_lbs, d_ubs_low, d_lbs_up, d_ubs, d_relu_neu[:, i], d_active_pattern[:, i, :],d_l1_lb, d_l1_ub,d_if_activation[i])
        for j in range(1, len(d_affine[0])):
            print(f"Affine {i}:{j} eq (LB,UB): ({d_lbs[init_id][j]}, {d_ubs[init_id][j]})")

def mergeBounds(d_lbsL,d_ubsL):
    d_lbs = cp.amax(d_lbsL, axis=0)
    d_ubs = cp.amin(d_ubsL, axis=0)
    #print(f"DEBUG -> d_ubsL: {d_ubsL}\n d_lbsL: {d_lbsL};\n d_lbs: {d_lbs}\n d_ubs:{d_ubs}")
    return d_lbs,d_ubs

def noPrintCondense(d_affine, if_activation,d_if_activation, d_l1_lb,d_l1_ub,domains,d_relu_dp,d_symb,d_relu_neu,d_active_pattern,NO_OF_INITIALS):
    d_lbsL = cp.zeros((len(domains), NO_OF_INITIALS,len(d_affine[0])))
    d_ubsL = cp.zeros((len(domains), NO_OF_INITIALS, len(d_affine[0])))
    for i in range(1, len(d_affine)):
        j = 0
        if ("DeepPoly" in domains):
            d_ineq_lte, d_ineq_gte = dpG.back_propagate_GPU(d_affine, d_relu_dp, i, if_activation)
            d_lbsL[j], d_ubsL[j] =   dpG.get_bounds_GPU(d_ineq_lte, d_ineq_gte, d_l1_lb, d_l1_ub)
            j += 1
        if ("Symbolic" in domains):
            d_ineq_lte, d_ineq_gte = symG.back_propagate_GPU(d_affine, d_symb, i, if_activation)
            d_lbsL[j], d_ubsL[j] = symG.get_bounds_GPU(d_ineq_lte, d_ineq_gte, d_l1_lb, d_l1_ub)
            j += 1
        if ("Neurify" in domains):
            d_ineq_lte, d_ineq_gte = neuG.back_propagate_GPU(d_affine, d_relu_neu, i,if_activation)
            d_lbs_low, d_ubs_low = neuG.get_bounds_GPU(d_ineq_lte, d_ineq_lte, d_l1_lb,d_l1_ub)
            d_lbs_up, d_ubs_up = neuG.get_bounds_GPU(d_ineq_gte, d_ineq_gte, d_l1_lb,d_l1_ub)
            d_lbsL[j], d_ubsL[j] = d_lbs_low,d_ubs_up
            j += 1
        d_lbs,d_ubs = mergeBounds(d_lbsL,d_ubsL)
        if (any(if_activation[i])):
            if ("DeepPoly" in domains):
                dpG.relu_compute_GPU(d_lbs, d_ubs, d_relu_dp[:, i], d_active_pattern[:, i],d_if_activation[i])
            if ("Symbolic" in domains):
                symG.relu_compute_GPU(d_lbs, d_ubs, d_symb[:,i], d_active_pattern[:,i], d_l1_lb, d_l1_ub,d_if_activation[i])
            if ("Neurify" in domains):
                neuG.relu_compute_GPU(d_lbs, d_ubs_low, d_lbs_up, d_ubs, d_relu_neu[:,i], d_active_pattern[:,i,:],d_l1_lb,d_l1_ub,d_if_activation[i])
    #print(f"DP:\n{d_active_pattern}")
    #print(f"SYM:\n{d_active_pattern_symb}")
    #print(f"NEU:\n{d_active_pattern_neu}")

def analyze(netGPU,l1_lbL,l1_ubL,percent,domains):
    d_affine, if_activation, d_if_activation, var_index, inv_var_index, outNodes, dims, l1_lb, l1_ub, sensitive, NO_OF_LAYERS, MAX_NODES_IN_LAYER = netGPU
    if (l1_lbL == None):
        l1_lbL = l1_lb
        l1_ubL = l1_ub

    l1_lb_list, l1_ub_list = commons.splitInitial(l1_lbL, l1_ubL, sensitive)
    s = ""
    for l1_lb in l1_lb_list:
        s += " + " + str(l1_lb.shape[0])
    print(Fore.CYAN+ f"\t\t\t\tIntial batches: {s}",Style.RESET_ALL)
    activatedL, deactivatedL, outcomeL = [], [], []
    for i in range(len(l1_ub_list)):
        if(len(l1_ub_list[i]) == 0):
            continue
        d_l1_lb = cp.asarray(l1_lb_list[i])
        d_l1_ub = cp.asarray(l1_ub_list[i])
        NO_OF_INITIALS = len(d_l1_lb)
        print(Fore.CYAN + f"\t\tNO_OF_LAYER:{NO_OF_LAYERS}; MAX_NODES_IN_LAYER:{MAX_NODES_IN_LAYER}; NO_OF_INITIALS:{NO_OF_INITIALS}",Style.RESET_ALL)

        d_active_pattern = cp.zeros((NO_OF_INITIALS, NO_OF_LAYERS + 1, MAX_NODES_IN_LAYER + 1))
        d_relu_dp,d_symb,d_relu_neu = None,None,None
        if ("DeepPoly" in domains):
            d_relu_dp = cp.zeros((NO_OF_INITIALS,NO_OF_LAYERS + 1, MAX_NODES_IN_LAYER + 1, 4))
        if ("Symbolic" in domains):
            d_symb = cp.zeros((NO_OF_INITIALS,NO_OF_LAYERS + 1, MAX_NODES_IN_LAYER + 1, 3))
        if ("Neurify" in domains):
            d_relu_neu = cp.zeros((NO_OF_INITIALS,NO_OF_LAYERS + 1, MAX_NODES_IN_LAYER + 1, 4))

        # Removes NumbaPerformanceWarning and others but slow down everything significantly.
        warnings.filterwarnings("ignore")
        #miniPrintCondense(d_affine, if_activation,d_if_activation, d_l1_lb,d_l1_ub,domains,d_relu_dp,d_symb,d_relu_neu,d_active_pattern,NO_OF_INITIALS)
        noPrintCondense(d_affine, if_activation,d_if_activation, d_l1_lb,d_l1_ub,domains,d_relu_dp,d_symb,d_relu_neu,d_active_pattern,NO_OF_INITIALS)

        #print(f"activation->{d_active_pattern}")
        outcome = oneOutput(d_affine,d_relu_dp,d_relu_neu,d_symb,if_activation,d_l1_lb,d_l1_ub,outNodes,inv_var_index,domains)
        active_pattern = cp.asnumpy(d_active_pattern)
        activated, deactivated = dpG.active_convert(active_pattern, dims, inv_var_index,outcome)
        activatedL.append(activated)
        deactivatedL.append(deactivated)
        outcomeL.append(outcome)
    factor = (len(l1_ub_list) - 1) * len(l1_ub_list[0]) + len(l1_ub_list[-1])
    #print(f"factor:{(factor / len(l1_lbL))}")
    percent = percent / (factor / len(l1_lbL))
    return activatedL, deactivatedL, outcomeL,l1_lb_list, l1_ub_list,percent,inv_var_index




