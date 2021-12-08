import numpy as np
import cupy as cp
from numba import cuda
from libra.core.cfg import Node, Function, Activation,Basic
from libra.core.expressions import VariableIdentifier
from libra.optimized.abstractDomainGPU import AbstractDomainGPU
from libra.optimized.deepPolyGPU import DeepPolyGPU
from libra.optimized.symbolicGPU import SymbolicGPU
from libra.optimized.neurifyGPU import NeurifyGPU
import warnings
from apronpy.var import PyVar

class ProductGPU(AbstractDomainGPU):
    def __init__(self):
        pass

    def miniPrintCondense(self,d_affine, if_activation,d_active_pattern, d_l1_lb,d_l1_ub,domains,d_relu_dp = None,d_symb = None,d_relu_neu=None, d_l1_lb_neu=None,d_l1_ub_neu=None):
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
            d_lbs, d_ubs = self.mergeBounds(d_lbsL, d_ubsL)
            if ("DeepPoly" in domains):
                dpG.relu_compute_GPU(d_lbs, d_ubs, d_relu_dp[i], d_active_pattern, d_l1_lb, d_l1_ub)
            if ("Symbolic" in domains):
                smbG.relu_compute_GPU(d_lbs, d_ubs, d_symb[i], d_active_pattern, d_l1_lb, d_l1_ub)
            if ("Neurify" in domains):
                neuG.relu_compute_GPU(d_lbs, d_ubs_low, d_lbs_up, d_ubs, d_relu_neu[i], d_active_pattern, d_l1_lb_neu,
                                      d_l1_ub_neu)

            for j in range(1, len(d_affine[0])):
                print(f"Affine {i}:{j} eq (LB,UB): ({d_lbs[j]}, {d_ubs[j]})")

    def mergeBounds(self,d_lbsL,d_ubsL):
        d_lbs = cp.amax(d_lbsL, axis=0)
        d_ubs = cp.amin(d_ubsL, axis=0)
        #print(f"DEBUG -> d_ubsL: {d_ubsL}\n d_lbsL: {d_lbsL};\n d_lbs: {d_lbs}\n d_ubs:{d_ubs}")
        return d_lbs,d_ubs

    def noPrintCondense(self,d_affine, if_activation,d_active_pattern, d_l1_lb,d_l1_ub,domains,d_relu_dp = None,d_symb = None,d_relu_neu=None, d_l1_lb_neu=None,d_l1_ub_neu=None):
        dpG = DeepPolyGPU()
        smbG = SymbolicGPU()
        neuG = NeurifyGPU()
        d_lbsL = cp.zeros((len(domains), len(d_affine[0])))
        d_ubsL = cp.zeros((len(domains), len(d_affine[0])))
        for i in range(1, len(d_affine)):
            j = 0
            if ("DeepPoly" in domains):
                d_lbsL[j], d_ubsL[j], ineq_lte, ineq_gte = dpG.back_propagate_GPU(d_affine, d_relu_dp, i, if_activation,d_active_pattern, d_l1_lb,d_l1_ub)
                j += 1
            if ("Symbolic" in domains):
                d_lbsL[j], d_ubsL[j], ineq_lte, ineq_gte = smbG.back_propagate_GPU(d_affine, d_symb, i, if_activation,d_active_pattern, d_l1_lb, d_l1_ub)
                j += 1
            if ("Neurify" in domains):
                d_lbs_low, d_ubs_low, d_lbs_up, d_ubs_up, ineq_lte, ineq_gte = neuG.back_propagate_GPU(d_affine, d_relu_neu, i,if_activation,d_active_pattern,d_l1_lb_neu,d_l1_ub_neu)
                d_lbsL[j], d_ubsL[j] = d_lbs_low,d_ubs_up
                j += 1
            d_lbs,d_ubs = self.mergeBounds(d_lbsL,d_ubsL)
            if (if_activation[i][1] == 1):
                if ("DeepPoly" in domains):
                    dpG.relu_compute_GPU(d_lbs, d_ubs, d_relu_dp[i], d_active_pattern[i], d_l1_lb, d_l1_ub)
                if ("Symbolic" in domains):
                    smbG.relu_compute_GPU(d_lbs, d_ubs, d_symb[i], d_active_pattern[i], d_l1_lb, d_l1_ub)
                if ("Neurify" in domains):
                    neuG.relu_compute_GPU(d_lbs, d_ubs_low, d_lbs_up, d_ubs, d_relu_neu[i], d_active_pattern[i], d_l1_lb_neu,d_l1_ub_neu)


    def network_condense_GPU(self,nodes, initial,domains):
        # equation[n1][n2] stores the bias and coeff of nodes of previous layer to form x[n1][n2] in order
        # if_activation[n1][n2] stores if there is an activation on x[n1][n2] (only relu considered for now)
        NO_OF_LAYERS, MAX_NODES_IN_LAYER = self.getNetShape(nodes)
        print(f"NO_OF_LAYER:{NO_OF_LAYERS}; MNIL:{MAX_NODES_IN_LAYER}")
        var_index: dict(str, (int, int)) = dict()
        row_id,col_id,flag = (0,1,False)

        affine = np.zeros((NO_OF_LAYERS + 1, MAX_NODES_IN_LAYER + 1, MAX_NODES_IN_LAYER + 1)).astype(np.float32)
        if_activation = np.zeros((NO_OF_LAYERS + 1, MAX_NODES_IN_LAYER + 1)).astype(np.float32)
        active_pattern = np.zeros((NO_OF_LAYERS + 1, MAX_NODES_IN_LAYER + 1)).astype(np.float32)
        l1_lb = np.zeros(MAX_NODES_IN_LAYER + 1).astype(np.float32)
        l1_ub = np.zeros(MAX_NODES_IN_LAYER + 1).astype(np.float32)
        dims = np.ones(NO_OF_LAYERS + 1).astype(np.int32)

        if ("DeepPoly" in domains):
            relu_dp = np.zeros((NO_OF_LAYERS + 1, MAX_NODES_IN_LAYER + 1, 4)).astype(np.float32)
        if ("Symbolic" in domains):
            symb = np.zeros((NO_OF_LAYERS + 1, MAX_NODES_IN_LAYER + 1, 3)).astype(np.float32)
        if ("Neurify" in domains):
            relu_neu = np.zeros((NO_OF_LAYERS + 1, MAX_NODES_IN_LAYER + 1, 4)).astype(np.float32)
            l1_lb_neu = np.zeros((MAX_NODES_IN_LAYER + 1, 2)).astype(np.float32)  # bounds for LOW
            l1_ub_neu = np.zeros((MAX_NODES_IN_LAYER + 1, 2)).astype(np.float32)  # bounds for UP

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


        self.fillInput(nodes, affine, dims, if_activation, var_index, MAX_NODES_IN_LAYER)
        inv_var_index = {v: k for k, v in var_index.items()}

        # can remove layer 0 as it has no inequations
        # All these print are for debug mode. Actual will only activation pattern.
        d_symb,d_relu_dp,d_relu_neu,d_l1_ub_neu,d_l1_lb_neu = (None,None,None,None,None)
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
            d_l1_ub_neu = cp.asarray(l1_ub_neu)
            d_l1_lb_neu = cp.asarray(l1_lb_neu)

        # Removes NumbaPerformanceWarning and others but slow down everything significantly.
        warnings.filterwarnings("ignore")
        self.miniPrintCondense(d_affine, if_activation,d_active_pattern, d_l1_lb,d_l1_ub,domains,d_relu_dp = d_relu_dp,d_symb = d_symb,d_relu_neu=d_relu_neu, d_l1_lb_neu=d_l1_lb_neu,d_l1_ub_neu=d_l1_ub_neu)
        #self.noPrintCondense(d_affine, if_activation,d_active_pattern, d_l1_lb,d_l1_ub,domains,d_relu_dp = d_relu_dp,d_symb = d_symb,d_relu_neu=d_relu_neu, d_l1_lb_neu=d_l1_lb_neu,d_l1_ub_neu=d_l1_ub_neu)

        # print(f"activation->{d_active_pattern}")
        #outcome = self.oneOutput(affine[-1], d_affine, d_relu, if_activation, d_l1_lb, d_l1_ub)
        #active_pattern = cp.asnumpy(d_active_pattern)
        #activated, deactivated = self.active_convert(active_pattern, dims, inv_var_index)
        # print(f"GPU active:{activated}; deactive:{deactivated}; outcome:{outcome}")
        #return activated, deactivated, outcome



