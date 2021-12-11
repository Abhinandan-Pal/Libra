import numpy as np
import cupy as cp
from numba import cuda
from libra.core.cfg import Node, Function, Activation,Basic
from libra.core.expressions import VariableIdentifier
from libra.optimized.abstractDomainGPU import AbstractDomainGPU
import warnings
from apronpy.var import PyVar

class NeurifyGPU(AbstractDomainGPU):
    def __init__(self):
        pass

    def get_bounds_GPU(self,d_l1_lte, d_l1_gte, d_l1_lb, d_l1_ub):
        @cuda.jit
        def bound_helper(l1_lte, l1_gte, l1_lb, l1_ub, lbs, ubs):
            id = cuda.grid(1)
            if (id >= len(lbs)):
                return
            lbs[id] = l1_lte[id, 0]
            for i in range(1, len(l1_lte[id])):
                if (l1_lte[id, i] < 0):
                    lbs[id] += l1_lte[id, i] * l1_ub[i][1]
                else:
                    lbs[id] += l1_lte[id, i] * l1_lb[i][0]
            ubs[id] = l1_gte[id, 0]
            for i in range(1, len(l1_gte[id])):
                if (l1_gte[id, i] > 0):
                    ubs[id] += l1_gte[id, i] * l1_ub[i][1]
                else:
                    ubs[id] += l1_gte[id, i] * l1_lb[i][0]

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

    def get_bounds_single_neurify(self,l1_layer_lte, l1_layer_gte, node_num: int, l1_lb, l1_ub, relu_val=[1, 0, 1, 0]):
        l1_lte = l1_layer_lte[node_num] * relu_val[0]

        l1_lte[0] += relu_val[1]

        l1_gte = l1_layer_gte[node_num] * relu_val[2]
        l1_gte[0] += relu_val[3]
        lb = l1_lte[0]
        for i in range(1, len(l1_lte)):
            if (l1_lte[i] < 0):
                lb += l1_lte[i] * l1_ub[i][0]
            else:
                lb += l1_lte[i] * l1_lb[i][0]
        ub = l1_gte[0]
        for i in range(1, len(l1_gte)):
            if (l1_gte[i] > 0):
                ub += l1_gte[i] * l1_ub[i][1]
            else:
                ub += l1_gte[i] * l1_lb[i][1]
        return lb, ub

    def relu_compute_GPU(self,d_lbs_low, d_ubs_low,d_lbs_up, d_ubs_up, d_relu_layer, d_active_pattern, d_l1_lb, d_l1_ub):
        @cuda.jit
        def relu_compute_helper(lbs_up, ubs_up, lbs_low, ubs_low, relu_layer, active_pattern):
            id = cuda.grid(1)
            if (id < 1 or id >= len(ubs_up)):
                return
            relu_layer[id][1] = 0.0
            if (ubs_low[id] < 0):
                relu_layer[id][0] = 0.0
                active_pattern[0][id] = 0
            elif (lbs_low[id] > 0):
                relu_layer[id][0] = 1.0
                active_pattern[0][id] = 1
            else:
                slope = ubs_low[id] / (ubs_low[id] - lbs_low[id])
                relu_layer[id][0] = slope
                active_pattern[0][id] = 2
            relu_layer[id][3] = 0.0
            if (ubs_up[id] < 0):
                relu_layer[id][2] = 0.0
                active_pattern[1][id] = 0
            elif (lbs_up[id] > 0):
                relu_layer[id][2] = 1.0
                active_pattern[1][id] = 1
            else:
                slope = ubs_up[id] / (ubs_up[id] - lbs_up[id])
                y_coeff = -ubs_up[id] * lbs_up[id] / (ubs_up[id] - lbs_up[id])
                relu_layer[id][2] = slope
                relu_layer[id][3] = y_coeff
                active_pattern[1][id] = 2

        print(f"d_lbs_up->{d_lbs_up} ; \nd_ubs_up->{d_ubs_up}")
        print(f"d_lbs_low->{d_lbs_low} ; \nd_ubs_low->{d_ubs_low}")
        tpb = (min(1024, len(d_lbs_low)),)
        bpg = (int(np.ceil(len(d_lbs_low) / tpb[0])),)
        relu_compute_helper[bpg, tpb](d_lbs_up,
                                      d_ubs_up,
                                      d_lbs_low,
                                      d_ubs_low,
                                      d_relu_layer,
                                      d_active_pattern)
        print(f"d_relu_layer->{d_relu_layer}")
        return d_relu_layer, d_active_pattern

    def back_affine_GPU(self,d_ineq_prev_lte, d_ineq_prev_gte, d_ln_coeff_lte, d_ln_coeff_gte):
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
                                         d_ineq_prev_gte)
        return d_l1_lte, d_l1_gte

    def back_relu_GPU(self,d_relu_layer, d_ln_coeff_lte, d_ln_coeff_gte):
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

    def back_propagate_GPU(self, d_affine, d_relu, layer: int, if_activation, d_active_pattern, d_l1_lb, d_l1_ub):
        # shift the CP creation to caller.
        d_ln_coeff_lte = d_affine[layer].copy().astype('float32')  # Need to create copies
        d_ln_coeff_gte = d_affine[layer].copy().astype('float32')
        layer_t = layer
        while (layer != 1):  # layer zero is input and layer one is in already in terms of input
            # First relu of previous layer
            if (if_activation[layer - 1][1] == True):
                self.back_relu_GPU(d_relu[layer - 1], d_ln_coeff_lte, d_ln_coeff_gte)
            # Then affine of previous layer
            d_ineq_prev_gte = d_affine[layer - 1]
            d_ineq_prev_lte = d_affine[layer - 1]
            d_ln_coeff_lte, d_ln_coeff_gte = self.back_affine_GPU(d_ineq_prev_lte, d_ineq_prev_gte, d_ln_coeff_lte,
                                                             d_ln_coeff_gte)
            layer -= 1

        d_lbs_up, d_ubs_up = self.get_bounds_GPU(d_ln_coeff_gte, d_ln_coeff_gte, d_l1_lb, d_l1_ub)
        d_lbs_low, d_ubs_low = self.get_bounds_GPU(d_ln_coeff_lte, d_ln_coeff_lte, d_l1_lb, d_l1_ub)

        '''if (if_activation[layer_t][1] == 1):
            self.relu_compute_GPU(d_ln_coeff_lte, d_ln_coeff_gte, d_relu[layer_t], d_active_pattern[:, layer_t, :],
                             d_l1_lb, d_l1_ub)
        else:
            pass
        # return d_active_pattern'''

        ''''# Different return for debug purposes'''
        ln_coeff_gte = cp.asnumpy(d_ln_coeff_gte).astype(np.float32)
        ln_coeff_lte = cp.asnumpy(d_ln_coeff_lte).astype(np.float32)
        return d_lbs_low, d_ubs_low,d_lbs_up, d_ubs_up, ln_coeff_lte, ln_coeff_gte

    # Simplification Assumption: output layer has 2 nodes.
    def oneOutput(self,last, d_affine, d_relu, if_activation, d_l1_lb, d_l1_ub):
        ln_coeff_lte = np.zeros(d_affine[1].shape).astype('float32')
        # ln_coeff_lte[1] = np.subtract(last[1], last[2])
        ln_coeff_lte[1][1] = 1
        ln_coeff_lte[1][2] = -1
        d_ln_coeff_lte = cp.asarray(ln_coeff_lte)
        d_ln_coeff_gte = d_ln_coeff_lte.copy().astype('float32')
        layer = len(d_affine)
        layer_t = layer
        # print(f"INITIAL :{ineq_str_direct(ln_coeff_lte[1], 4, 1, '>=', 3)}")

        while (layer != 1):  # layer zero is input and layer one is in already in terms of input
            # First relu of previous layer
            if (if_activation[layer - 1][1] == True):
                self.back_relu_GPU(d_relu[layer - 1], d_ln_coeff_lte, d_ln_coeff_gte)
            # Then affine of previous layer
            d_ineq_prev_gte = d_affine[layer - 1]
            d_ineq_prev_lte = d_affine[layer - 1]
            d_ln_coeff_lte, d_ln_coeff_gte = self.back_affine_GPU(d_ineq_prev_lte, d_ineq_prev_gte, d_ln_coeff_lte,
                                                             d_ln_coeff_gte)
            layer -= 1

        d_lbs_low, d_ubs_low = self.get_bounds_GPU(d_ln_coeff_lte, d_ln_coeff_lte, d_l1_lb, d_l1_ub)
        d_lbs_up, d_ubs_up = self.get_bounds_GPU(d_ln_coeff_gte, d_ln_coeff_gte, d_l1_lb, d_l1_ub)
        lbs_low = cp.asnumpy(d_lbs_low)
        ubs_low = cp.asnumpy(d_ubs_low)
        lbs_up = cp.asnumpy(d_lbs_up)
        ubs_up = cp.asnumpy(d_ubs_up)

        '''
        ln_coeff_gte = cp.asnumpy(d_ln_coeff_gte).astype(np.float32)
        ln_coeff_lte = cp.asnumpy(d_ln_coeff_lte).astype(np.float32)
        print(f"DEBUG --> l1_lb: {d_l1_lb}; l1_ub:{d_l1_ub}")
        print(f" eq LTE L1: {ineq_str_direct(ln_coeff_lte[1], 4, 1, '>=', 0)}")
        print(f" eq GTE L1: {ineq_str_direct(ln_coeff_gte[1], 4, 1, '<=', 0)}")
        print(f"DEBUG --> lbs:{d_lbs}; ubs:{d_ubs}")'''
        if (lbs_low[1] > 0.0 and ubs_up[1] > 0.0):
            stmt = "x" + str(len(d_affine) - 1) + str(1)
            return VariableIdentifier(stmt)
        elif (ubs_low[1] < 0.0 and ubs_up[1] < 0.0):
            stmt = "x" + str(len(d_affine) - 1) + str(2)
            return VariableIdentifier(stmt)
        else:
            return None

    def active_convert(self,active_status, dims, inv_var_index):
        activated = set()
        deactivated = set()
        node_num = 3
        for l_id in range(1, len(dims[1:])):
            for n_id in range(1, dims[l_id]):
                if (active_status[0, l_id, n_id] == 0 and active_status[1, l_id, n_id] == 0):
                    stmt = inv_var_index[(l_id, n_id)]
                    val = Basic(node_num, [PyVar(stmt)])
                    deactivated.add(val)
                elif (active_status[0, l_id, n_id] == 1 and active_status[1, l_id, n_id] == 1):
                    stmt = inv_var_index[(l_id, n_id)]
                    val = Basic(node_num, [PyVar(stmt)])
                    activated.add(val)
                node_num += 1
            node_num += 1
        return activated, deactivated

    def getNetShape(self,nodes):
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
        return NO_OF_LAYERS, MAX_NODES_IN_LAYER

    def fillInput(self,nodes,affine,dims,if_activation,var_index,MNIL):
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
                    eq = self.texpr_to_dict(eq)
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

    def detailedPrintCondense(self,d_affine,d_relu,d_active_pattern,d_l1_lb,d_l1_ub,if_activation,relu,var_index,inv_var_index,l1_lb,l1_ub):
        print(f"var_index = {var_index}")
        print(f"inv_var_index = {inv_var_index}")

        for i in range(1, len(d_affine)):

            print(f"\t\t LAYER {i} Input Equations")
            for j in range(1, len(d_affine[0])):
                print(
                    f"Node: {j} -> if_activation: {if_activation[i, j]}\n eq: {self.ineq_str(d_affine[i, j], i, j, '=', i - 1, inv_var_index)} ")
            d_lbs_low, d_ubs_low,d_lbs_up, d_ubs_up,ineq_lte, ineq_gte = self.back_propagate_GPU(d_affine, d_relu, i, if_activation, d_active_pattern, d_l1_lb,
                                                         d_l1_ub)
            if (if_activation[i][1] == 1):
                self.relu_compute_GPU(d_lbs_low, d_ubs_low,d_lbs_up, d_ubs_up, d_relu[i], d_active_pattern[:,i,:],d_l1_lb,d_l1_ub)
            relu[i] = cp.asnumpy(d_relu[i])
            print(f"\t\t LAYER {i} Substituted")
            for j in range(1, len(d_affine[0])):
                print(f"\tNode {j}")
                print(f" eq LTE L1: {self.ineq_str(ineq_lte[j], i, j, '>=', 0, inv_var_index)}")
                print(f" eq GTE L1: {self.ineq_str(ineq_gte[j], i, j, '<=', 0, inv_var_index)}")
                print(f" eq LOW (LB,UB): {self.get_bounds_single_neurify(ineq_lte, ineq_lte, j, l1_lb, l1_ub)}")
                print(f" eq UP (LB,UB): {self.get_bounds_single_neurify(ineq_gte, ineq_gte, j, l1_lb, l1_ub)}")
            if (if_activation[i, 1] == 1):  # assuming if first node in a layer has activation then all do
                print(f"\t RELU-LAYER {i}")
                for j in range(1, len(d_affine[0])):
                    print(f"\tNode {j}")
                    print(f" Relu eq LTE: Slope: {relu[i][j][0]}, Y-Coeff: {relu[i][j][1]}")
                    print(f" Relu eq GTE: Slope: {relu[i][j][2]}, Y-Coeff: {relu[i][j][3]}")
                    relu_val = [relu[i][j][0], relu[i][j][1], relu[i][j][0], relu[i][j][1]]
                    print(
                        f"Relu eq LOW (LB,UB): {self.get_bounds_single_neurify(ineq_lte, ineq_lte, j, l1_lb, l1_ub, relu_val=relu_val)}")
                    relu_val = [relu[i][j][2], relu[i][j][3], relu[i][j][2], relu[i][j][3]]
                    print(
                        f"Relu eq UP (LB,UB): {self.get_bounds_single_neurify(ineq_gte, ineq_gte, j, l1_lb, l1_ub, relu_val=relu_val)}")
            # print stuff
            else:
                print(f"\t\t NO RELU ON LAYER {i}")
        #print(f"activation->{d_active_pattern}")

    def miniPrintCondense(self, d_affine, d_relu, d_active_pattern, d_l1_lb, d_l1_ub, if_activation, l1_lb, l1_ub,relu):
        for i in range(1, len(d_affine)):
            d_lbs_low, d_ubs_low, d_lbs_up, d_ubs_up, ineq_lte, ineq_gte = self.back_propagate_GPU(d_affine, d_relu, i,
                                                                                                   if_activation,
                                                                                                   d_active_pattern,
                                                                                                   d_l1_lb,
                                                                                                   d_l1_ub)
            if (if_activation[i][1] == 1):
                self.relu_compute_GPU(d_lbs_low, d_ubs_low,d_lbs_up, d_ubs_up, d_relu[i], d_active_pattern[:,i,:],d_l1_lb,d_l1_ub)
            relu[i] = cp.asnumpy(d_relu[i])

            if (if_activation[i, 1] == 1):  # assuming if first node in a layer has activation then all do
                for j in range(1, len(d_affine[0])):
                    relu_val = [relu[i][j][0], relu[i][j][1], relu[i][j][0], relu[i][j][1]]
                    print(
                        f"Relu{i}{j} eq LOW (LB,UB): {self.get_bounds_single(ineq_lte, ineq_lte, j, l1_lb, l1_ub, relu_val=relu_val)}")
                    relu_val = [relu[i][j][2], relu[i][j][3], relu[i][j][2], relu[i][j][3]]
                    print(
                        f"Relu{i}{j} eq UP (LB,UB): {self.get_bounds_single_neurify(ineq_gte, ineq_gte, j, l1_lb, l1_ub, relu_val=relu_val)}")
            else:
                for j in range(1, len(d_affine[0])):
                    print(
                        f" eq{i}{j} LOW (LB,UB): {self.get_bounds_single_neurify(ineq_lte, ineq_lte, j, l1_lb, l1_ub)}")
                    print(
                        f" eq{i}{j} UP (LB,UB): {self.get_bounds_single_neurify(ineq_gte, ineq_gte, j, l1_lb, l1_ub)}")

    def noPrintCondense(self,d_affine, d_relu, i, if_activation,d_active_pattern, d_l1_lb,d_l1_ub):
        for i in range(1, len(d_affine)):
            d_lbs_low, d_ubs_low, d_lbs_up, d_ubs_up, ineq_lte, ineq_gte = self.back_propagate_GPU(d_affine, d_relu, i,
                                                                                                   if_activation,
                                                                                                   d_active_pattern,
                                                                                                   d_l1_lb,
                                                                                                   d_l1_ub)
            if (if_activation[i][1] == 1):
                self.relu_compute_GPU(d_lbs_low, d_ubs_low,d_lbs_up, d_ubs_up, d_relu[i], d_active_pattern[:,i,:],d_l1_lb,d_l1_ub)
    def network_condense_GPU(self,nodes, initial):
        # equation[n1][n2] stores the bias and coeff of nodes of previous layer to form x[n1][n2] in order
        # if_activation[n1][n2] stores if there is an activation on x[n1][n2] (only relu considered for now)
        NO_OF_LAYERS, MAX_NODES_IN_LAYER = self.getNetShape(nodes)
        print(f"NO_OF_LAYER:{NO_OF_LAYERS}; MNIL:{MAX_NODES_IN_LAYER}")

        var_index: dict(str, (int, int)) = dict()
        row_id,col_id,flag = (0,1,False)

        affine = np.zeros((NO_OF_LAYERS + 1, MAX_NODES_IN_LAYER + 1, MAX_NODES_IN_LAYER + 1)).astype(np.float32)
        relu = np.zeros((NO_OF_LAYERS + 1, MAX_NODES_IN_LAYER + 1, 4)).astype(np.float32)
        if_activation = np.zeros((NO_OF_LAYERS + 1, MAX_NODES_IN_LAYER + 1)).astype(np.float32)
        active_pattern = np.zeros((2, NO_OF_LAYERS + 1, MAX_NODES_IN_LAYER + 1)).astype(np.float32)
        l1_lb = np.zeros((MAX_NODES_IN_LAYER + 1, 2)).astype(np.float32)  # bounds for LOW
        l1_ub = np.zeros((MAX_NODES_IN_LAYER + 1, 2)).astype(np.float32)  # bounds for UP
        dims = np.ones(NO_OF_LAYERS + 1).astype(np.int32)

        # obtain the lower bound and upper bound for input layer using "initial"
        # Assuming "initial" contains input from 0 to nth input in order.
        i = 1
        for var, bound in initial.bounds.items():
            '''if(type(bound) == VariableIdentifier):
                l1_lb[i] = bound.lower
                l1_ub[i] = bound.upper
            else:'''
            l1_lb[i][0] = bound[0].lower
            l1_lb[i][1] = bound[0].upper
            l1_ub[i][0] = bound[1].upper
            l1_ub[i][1] = bound[1].upper
            var_index[str(var)] = (row_id, col_id)
            col_id += 1
            i += 1
        #print(f"\tDEBUG ---> l1_lb: {l1_lb} \n l1_ub: {l1_ub}")

        self.fillInput(nodes, affine, dims, if_activation, var_index, MAX_NODES_IN_LAYER)
        inv_var_index = {v: k for k, v in var_index.items()}

        # can remove layer 0 as it has no inequations
        # All these print are for debug mode. Actual will only activation pattern.
        d_affine = cp.asarray(affine)
        d_relu = cp.asarray(relu)
        d_active_pattern = cp.asarray(active_pattern)
        d_l1_lb = cp.asarray(l1_lb)
        d_l1_ub = cp.asarray(l1_ub)
        # Removes NumbaPerformanceWarning and others but slow down everything significantly.
        warnings.filterwarnings("ignore")
        self.detailedPrintCondense(d_affine,d_relu,d_active_pattern,d_l1_lb,d_l1_ub,if_activation,relu,var_index,inv_var_index,l1_lb,l1_ub)
        #self.miniPrintCondense(d_affine, d_relu, d_active_pattern, d_l1_lb, d_l1_ub, if_activation, l1_lb, l1_ub, relu)
        #self.noPrintCondense(d_affine, d_relu, i, if_activation, d_active_pattern, d_l1_lb, d_l1_ub)

        outcome = self.oneOutput(affine[-1], d_affine, d_relu, if_activation, d_l1_lb, d_l1_ub)
        active_pattern = cp.asnumpy(d_active_pattern)

        activated, deactivated = self.active_convert(active_pattern, dims, inv_var_index)
        print(f"GPU active:{activated}; deactive:{deactivated}; outcome:{outcome}")
        return activated, deactivated, outcome