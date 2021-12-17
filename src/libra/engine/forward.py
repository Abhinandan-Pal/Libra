"""
Forward Analysis Engine
=======================

:Author: Caterina Urban
"""
import time
import numpy as np
import cupy as cp
from numba import cuda

import copy
from copy import deepcopy
from queue import Queue
from pip._vendor.colorama import Fore, Style

from apronpy.manager import PyManager

from libra.core.statements import Call
from libra.engine.interpreter import Interpreter
from libra.semantics.forward import DefaultForwardSemantics
from libra.abstract_domains.state import State
from libra.core.cfg import Node, Function, Activation

from  libra.optimized import deeppoly_gpu
from libra.optimized.deepPolyGPU import DeepPolyGPU
from libra.optimized.symbolicGPU import SymbolicGPU
from libra.optimized.neurifyGPU import NeurifyGPU
from libra.optimized.productGPU import ProductGPU

from  libra.optimized import symbolic_gpu
from  libra.optimized import neurify_gpu
from  libra.optimized import product_gpu
from libra.optimized.deeppoly_cpu import network_condense_CPU

from apronpy.texpr0 import TexprRtype, TexprRdir, TexprDiscr, TexprOp
from apronpy.texpr1 import PyTexpr1
from apronpy.var import PyVar


class ForwardInterpreter(Interpreter):
    """Forward control flow graph interpreter."""

    def __init__(self, cfg, manager: PyManager, semantics, log=False):
        """Forward control flow graph interpreter construction.

        :param cfg: control flow graph to analyze
        :param semantics: semantics of statements in the control flow graph
        :param precursory: precursory control flow graph interpreter
        """
        super().__init__(cfg, semantics)
        self.manager = manager
        self._log = log

    def _state_log(self, state, outputs, full=True):
        """log of the state bounds (usually only Input/Output) of the state after a forward analysis step

        :param state: state of the analsis after a forward application
        :param outputs: set of outputs name
        :param full: True for full print or False for just Input/Output (Default False)
        """
        if self._log:
            input_color = Fore.YELLOW
            output_color = Fore.MAGENTA
            mid_color = Fore.LIGHTBLACK_EX
            error_color = Fore.RED
            outputs = {k.name for k in outputs}

            print("Forward Analysis (", Style.RESET_ALL, end='', sep='')
            print(input_color + "Input", Style.RESET_ALL, end='', sep='')
            print("|", Style.RESET_ALL, end='', sep='')
            if full:
                print(mid_color + "Hidden", Style.RESET_ALL, end='', sep='')
                print("|", Style.RESET_ALL, end='', sep='')

            print(output_color + "Output", Style.RESET_ALL, end='', sep='')
            print("): {", Style.RESET_ALL)

            if hasattr(state, "bounds") and isinstance(state.bounds, dict):
                inputs = [f"  {k} -> {state.bounds[k]}" for k in state.inputs]
                inputs.sort()
                print(input_color + "\n".join(inputs), Style.RESET_ALL)
                if full:
                    mid_states = [f"  {k} -> {state.bounds[k]}" for k in state.bounds.keys() - state.inputs - outputs]
                    mid_states.sort()
                    print(mid_color + "\n".join(mid_states), Style.RESET_ALL)
                outputs = [f"  {k} -> {state.bounds[k]}" for k in outputs]
                outputs.sort()
                print(output_color + "\n".join(outputs), Style.RESET_ALL)
            else:
                print(error_color + "Unable to show bounds on the param 'state'" +
                      "\n  > missing attribute 'state.bounds', or 'state.bounds' is not a dictionary" +
                      "\n  > next state logs will be hidden", Style.RESET_ALL)
                self._log = True

            print("}", Style.RESET_ALL)

    def texpr_to_dict(self, texpr):

        def do(texpr0, env):
            if texpr0.discr == TexprDiscr.AP_TEXPR_CST:
                result = dict()
                t0 = '{}'.format(texpr0.val.cst)
                t1 = eval(t0)
                t2 = str(t1)
                t3 = float(t2)
                result['_'] = t3
                return result
            elif texpr0.discr == TexprDiscr.AP_TEXPR_DIM:
                result = dict()
                result['{}'.format(env.var_of_dim[texpr0.val.dim.value].decode('utf-8'))] = 1.0
                return result
            else:
                assert texpr0.discr == TexprDiscr.AP_TEXPR_NODE
                left = do(texpr0.val.node.contents.exprA.contents, env)
                op = texpr0.val.node.contents.op
                if texpr0.val.node.contents.exprB:
                    right = do(texpr0.val.node.contents.exprB.contents, env)
                if op == TexprOp.AP_TEXPR_ADD:
                    result = deepcopy(left)
                    for var, val in right.items():
                        if var in result:
                            result[var] += val
                        else:
                            result[var] = val
                    return result
                elif op == TexprOp.AP_TEXPR_SUB:
                    result = deepcopy(left)
                    for var, val in right.items():
                        if var in result:
                            result[var] -= val
                        else:
                            result[var] = -val
                    return result
                elif op == TexprOp.AP_TEXPR_MUL:
                    # print('multiplying')
                    # print('left: ', left)
                    # print('right: ', right)
                    result = dict()
                    if '_' in left and len(left) == 1:
                        for var, val in right.items():
                            result[var] = left['_'] * right[var]
                    elif '_' in right and len(right) == 1:
                        for var, val in left.items():
                            result[var] = right['_'] * left[var]
                    else:
                        assert False
                    # print('result: ', result)
                elif op == TexprOp.AP_TEXPR_NEG:
                    result = deepcopy(left)
                    for var, val in result.items():
                        result[var] = -val
            return result

        texpr1 = texpr.texpr1.contents
        return do(texpr1.texpr0.contents, texpr1.env.contents)

    '''lte stores equation < Xn gte store equation > Xn'''
    '''If Previous layer is expressed in inequality form of x01 and x02  back_propagate_l1 expresses a node of 
    current layer in inequality form of x01 and x02 '''
    def back_propagate_l1(self, eq_layer: list[list[float]], ineq_prev_lte: list[list[float]],
                           ineq_prev_gte: list[list[float]], node_num: int) -> tuple[list[float],list[float]]:
        ln_coeff = eq_layer[node_num]  # store the bias term
        l1_lte = [0] * len(ineq_prev_lte[1])
        l1_gte = [0] * len(ineq_prev_lte[1])
        l1_lte[0] = ln_coeff[0]
        l1_gte[0] = ln_coeff[0]
        for i in range(1, len(l1_lte)):
            for p in range(0, len(ineq_prev_lte[1])):
                if(ln_coeff[i]>0):
                    l1_lte[p] += ln_coeff[i] * ineq_prev_lte[i][p]
                    l1_gte[p] += ln_coeff[i] * ineq_prev_gte[i][p]
                else:

                    l1_lte[p] += ln_coeff[i] * ineq_prev_gte[i][p]
                    l1_gte[p] += ln_coeff[i] * ineq_prev_lte[i][p]

        return l1_lte, l1_gte

    def back_propagate_l1_GPU(self,ln_coeff, ineq_prev_lte,
                              ineq_prev_gte):
        @cuda.jit
        def back_propagate_helper(l1_lte, l1_gte, coeff, ineq_prev_lte_i, ineq_prev_gte_i):
            k, p = cuda.grid(2)
            if (k >= len(coeff) or k >= len(l1_lte) or k >= len(l1_gte) or p >= len(ineq_prev_lte_i) or p >= len(
                    ineq_prev_gte_i)):
                return
            if (coeff[k] > 0):  # add check if k and p are valid
                l1_lte[k][p] += coeff[k] * ineq_prev_lte_i[p]  # should it be i or i-1?
                l1_gte[k][p] += coeff[k] * ineq_prev_gte_i[p]
            else:
                l1_lte[k][p] += coeff[k] * ineq_prev_gte_i[p]
                l1_gte[k][p] += coeff[k] * ineq_prev_lte_i[p]

        l1_lte = np.zeros((len(ln_coeff), len(ineq_prev_lte[1])))
        l1_gte = np.zeros((len(ln_coeff), len(ineq_prev_lte[1])))
        l1_lte[:, 0] = ln_coeff[:, 0]
        l1_gte[:, 0] = ln_coeff[:, 0]
        d_l1_lte = cp.array(l1_lte)
        d_l1_gte = cp.array(l1_gte)

        cuda_iters = (len(ln_coeff), len(ineq_prev_lte[1]))
        tpb = (min(32, cuda_iters[0]), min(32, cuda_iters[1]))
        bpg = (int(np.ceil(cuda_iters[0] / tpb[0])), int(np.ceil(cuda_iters[1] / tpb[1])))
        #print(f"tpb = {tpb}, bpg = {bpg}")
        for i in range(1, len(d_l1_lte)):
            d_coeff = cp.array(ln_coeff[:, i])

            d_ineq_prev_lte_i = cp.asarray(ineq_prev_lte[i])  # only load whats needed now to prevent out of memory

            d_ineq_prev_gte_i = cp.asarray(ineq_prev_gte[i])

            back_propagate_helper[bpg, tpb](d_l1_lte,
                                            d_l1_gte,
                                            d_coeff,
                                            d_ineq_prev_lte_i,
                                            d_ineq_prev_gte_i, )
            # pycuda.driver.Context.synchronize();
        l1_gte = cp.asnumpy(d_l1_gte)
        l1_lte = cp.asnumpy(d_l1_lte)
        return l1_lte, l1_gte

    '''Given a node of a layer expressed in inequality form of x01 and x02. it gives the upper and lower bound
     assuming x0n are in [0,1]'''
    def get_bounds_single(self,l1_layer_lte: list[list[float]],l1_layer_gte: list[list[float]],node_num:int , l1_lb = -1,l1_ub = 1):
        l1_lte = l1_layer_lte[node_num]
        l1_gte = l1_layer_gte[node_num]
        lb = l1_lte[0]
        for coeff in l1_lte[1:]:
            if (coeff < 0):
                lb += coeff * l1_ub
            else:
                lb += coeff* l1_lb
        ub = l1_gte[0]
        for coeff in l1_gte[1:]:
            if (coeff > 0):
                ub += coeff * l1_ub
            else:
                ub += coeff* l1_lb
        return lb, ub



    '''If current layer is expressed in inequality form of x01 and x02  relu_propagate_l1 expresses a node of
        current layer after relu in inequality form of x01 and x02 '''
    def relu_propagate_l1(self,l1_layer_lte:list[list[float]],l1_layer_gte: list[list[float]],node_num:int):
        l1_lte = l1_layer_lte[node_num]
        l1_gte = l1_layer_gte[node_num]
        lb,ub = self.get_bounds_single(l1_layer_lte,l1_layer_gte,node_num)
        l1_relu_lte = [0] * len(l1_lte)
        l1_relu_gte = [0] * len(l1_gte)
        '''Case 1(Strictly Negative)'''
        if(ub<0):
            return l1_relu_lte,l1_relu_gte
        '''Case 2(Strictly Positive)'''
        if(lb > 0):
            return l1_lte, l1_gte
        '''Case 3(Crossing Relu)'''

        slope = ub/(ub-lb)
        y_coeff = -ub*lb/(ub-lb)
        for i in range(len(l1_gte)):
            l1_relu_gte[i] = slope*l1_gte[i]
        l1_relu_gte[0]+= y_coeff
        b3_area = abs(ub*(ub-lb))
        c3_area = abs(lb*(ub-lb))
        if(c3_area < b3_area):
            for i in range(len(l1_lte)):
                l1_relu_lte[i] = l1_lte[i]
        return l1_relu_lte,l1_relu_gte

    def get_bounds_GPU(self,d_l1_lte, d_l1_gte, l1_lb=0, l1_ub=1):
        @cuda.jit
        def bound_helper(l1_lte, l1_gte, l1_lb, l1_ub, lbs, ubs):
            id = cuda.grid(1)
            if (id >= len(lbs)):
                return
            lbs[id] = l1_lte[id,0]
            for coeff in l1_lte[id, 1:]:
                if (coeff < 0):
                    lbs[id] += coeff * l1_ub[0]
                else:
                    lbs[id] += coeff * l1_lb[0]
            ubs[id] = l1_gte[id,0]
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

    def relu_propagate_l1_GPU(self,l1_lte, l1_gte):
        @cuda.jit
        def relu_prop_helper(l1_lte, l1_gte, lbs, ubs, l1_relu_lte, l1_relu_gte):
            id = cuda.grid(1)
            if (id >= len(ubs)):
                return
            if (ubs[id] < 0):
                return  # as initialized with zeros
            if (lbs[id] > 0):
                for i in range(len(l1_lte[id])):
                    l1_relu_lte[id][i] = l1_lte[id][i]
                    l1_relu_gte[id][i] = l1_gte[id][i]
                return
            slope = ubs[id] / (ubs[id] - lbs[id])
            y_coeff = -ubs[id] * lbs[id] / (ubs[id] - lbs[id])
            for j in range(len(l1_lte[id])):
                l1_relu_gte[id][j] = slope * l1_gte[id][j]
            b3_area = abs(ubs[id] * (ubs[id] - lbs[id]))
            c3_area = abs(lbs[id] * (ubs[id] - lbs[id]))
            l1_relu_gte[id][0] += y_coeff
            if (c3_area < b3_area):
                for i in range(len(l1_lte)):
                    l1_relu_lte[id][i] = l1_lte[id][i]


        d_l1_lte = cp.asarray(l1_lte)
        d_l1_gte = cp.asarray(l1_gte)
        d_lbs, d_ubs = self.get_bounds_GPU(d_l1_lte, d_l1_gte)
        d_l1_relu_lte = cp.zeros(l1_lte.shape)
        d_l1_relu_gte = cp.zeros(l1_lte.shape)
        tpb = (min(1024, len(l1_lte)),)
        bpg = (int(np.ceil(len(l1_lte) / tpb[0])),)
        relu_prop_helper[bpg, tpb](d_l1_lte,
                                   d_l1_gte,
                                   d_lbs,
                                   d_ubs,
                                   d_l1_relu_lte,
                                   d_l1_relu_gte)
        l1_relu_gte = cp.asnumpy(d_l1_relu_gte)
        l1_relu_lte = cp.asnumpy(d_l1_relu_lte)
        return l1_relu_lte, l1_relu_gte

    '''Add Case-3 and perform speed test. Might be significantly slower now'''
    def relu_propagate_l1_GPU2(self,l1_lte, l1_gte):
        @cuda.jit
        def relu_prop_helper2(l1_lte, l1_gte, lbs, ubs, l1_relu_lte, l1_relu_gte):
            idx, idy = cuda.grid(2)
            if (idx >= len(ubs) or idy >= len(l1_lte[idx])):
                return
            if (ubs[idx] < 0):
                return  # as initialized with zeros
            if (lbs[idx] > 0):
                l1_relu_lte[idx][idy] = l1_lte[idx][idy]
                l1_relu_gte[idx][idy] = l1_gte[idx][idy]
                return
            slope = ubs[idx] / (ubs[idx] - lbs[idx])
            l1_relu_gte[idx][idy] = slope * l1_gte[idx][idy]
            if (idy == 0):
                y_coeff = -ubs[idx] * lbs[idx] / (ubs[idx] - lbs[idx])
                l1_relu_gte[idx][0] += y_coeff
            b3_area = abs(ubs[idx] * (ubs[idx] - lbs[idx]))
            c3_area = abs(lbs[idx] * (ubs[idx] - lbs[idx]))
            if (c3_area < b3_area):
                l1_relu_lte[idx][idy] = l1_lte[idx][idy]

        d_l1_lte = cp.asarray(l1_lte)
        d_l1_gte = cp.asarray(l1_gte)
        d_lbs, d_ubs = self.get_bounds_GPU(d_l1_lte, d_l1_gte)
        d_l1_relu_lte = cp.zeros(l1_lte.shape)
        d_l1_relu_gte = cp.zeros(l1_lte.shape)
        tpb = (min(32, len(l1_lte)), min(32, len(l1_lte[0])))
        bpg = (int(np.ceil(len(l1_lte) / tpb[0])), int(np.ceil(len(l1_lte[0]) / tpb[0])))
        relu_prop_helper2[bpg, tpb](d_l1_lte,
                                    d_l1_gte,
                                    d_lbs,
                                    d_ubs,
                                    d_l1_relu_lte,
                                    d_l1_relu_gte)
        l1_relu_gte = cp.asnumpy(d_l1_relu_gte)
        l1_relu_lte = cp.asnumpy(d_l1_relu_lte)
        return l1_relu_lte, l1_relu_gte

    def ineq_str(self,ineq:list[float],layer_lhs,node_num,op,layer_rhs):
        if(op==">="):
            str_ineq = f"X{layer_lhs}{node_num} >= "
        elif (op == "="):
            str_ineq = f"X{layer_lhs}{node_num} = "
        else:
            str_ineq = f"X{layer_lhs}{node_num} <= "
        for i in range(len(ineq)):
            if(i==0):
                str_ineq += f"+({ineq[0]})"
            else:
                str_ineq += f"+({ineq[i]} * X{layer_rhs}{i})"
        return str_ineq


    def network_condense_CPU(self, nodes):
        # equation[n1][n2] stores the bias and coeff of nodes of previous layer to form x[n1][n2] in order
        # if_activation[n1][n2] stores if there is an activation on x[n1][n2] (only relu considered for now)
        ineq_lte: list[list[list[float]]] = []
        ineq_gte: list[list[list[float]]] = []
        ineq_relu_lte: list[list[list[float]]] = []
        ineq_relu_gte: list[list[list[float]]] = []
        if_activation: list[list[bool]] = []                    #TO-DO: can make the assumption if one node in a layer has relu then all do
        for i in range(3 + 1):  # NO OF LAYERS +1 as start from X11
            ineq1 = []
            ineq2 = []
            ineq3 = []
            ineq4 = []

            if_act = []
            for j in range(2 + 1):  # Max no of nodes in a layer +1 as start from X11
                ineq1.append([0.0])
                ineq2.append([0.0])
                ineq3.append([0.0])
                ineq4.append([0.0])
                if_act.append(False)
            ineq_lte.append(ineq1)
            ineq_gte.append(ineq2)
            ineq_relu_lte.append(ineq3)
            ineq_relu_gte.append(ineq4)
            if_activation.append(if_act)
        for current in nodes:
            if isinstance(current, Function):
                for (node, eq) in zip(current.stmts[0], current.stmts[1]):
                    size_of_previous_layer = 2 + 1  # TO-DO: get the actual value from cfg +1 for bias
                    coeffs = [0] * size_of_previous_layer
                    eq = self.texpr_to_dict(eq)
                    for var, val in eq.items():
                        if var != '_':
                            var = int(
                                var[2])  # TO-DO: do something more general than assume format X[N][N] for var name
                            coeffs[var] = val
                        else:
                            coeffs[0] = val
                    ineq_lte[int(str(node)[1])][int(str(node)[2])]= coeffs  # TO-DO: do something more general than assume format X[N][N] for var name
            elif isinstance(current, Activation):
                if_activation[int(str(current.stmts)[1])][int(str(current.stmts)[2])] = True
            else:
                pass
                '''     What to do here
                for stmt in reversed(current.stmts):
                    state = self.semantics.assume_call_semantics(stmt, state, self.manager)'''
        print(f'{ineq_lte}')
        for i in range(1, len(ineq_lte)):
            for j in range(1, len(ineq_lte[0])):
                print(f"\t\tX{i}{j}")
                print(f" if_activation: {if_activation[i][j]}\n eq: {self.ineq_str(ineq_lte[i][j],i,j,'=',i-1)} ")
                if(i != 1):
                    ineq_lte[i][j],ineq_gte[i][j] = self.back_propagate_l1(ineq_lte[i],ineq_relu_lte[i-1],ineq_relu_gte[i-1], j)
                else:
                    ineq_gte[i][j] = copy.deepcopy(ineq_lte[i][j])
                print(f" eq LTE L1: {self.ineq_str(ineq_lte[i][j],i,j,'>=',0)}")                                #Use deep copy or on relu values are changing
                print(f" eq GTE L1: {self.ineq_str(ineq_gte[i][j], i, j, '<=', 0)}")
                print(f"eq (LB,UB): {self.get_bounds_single(ineq_lte[i],ineq_gte[i],j)}")

                if(if_activation[i][j]):
                    ineq_relu_lte[i][j], ineq_relu_gte[i][j] = self.relu_propagate_l1(ineq_lte[i],ineq_gte[i],j)
                else:
                    ineq_relu_lte[i][j], ineq_relu_gte[i][j] = ineq_lte[i][j], ineq_gte[i][j]

                print(f" Relu eq LTE L1: {self.ineq_str(ineq_relu_lte[i][j], i, j, '>=', 0)}")
                print(f" Relu eq GTE L1: {self.ineq_str(ineq_relu_gte[i][j], i, j, '<=', 0)}")
                print(f"Relu eq (LB,UB): {self.get_bounds_single(ineq_relu_lte[i], ineq_relu_gte[i], j)}")

    def network_condense_GPU(self, nodes):
        # equation[n1][n2] stores the bias and coeff of nodes of previous layer to form x[n1][n2] in order
        # if_activation[n1][n2] stores if there is an activation on x[n1][n2] (only relu considered for now)
        NO_OF_LAYERS = 3
        MAX_NODES_IN_LAYER = 2
        ineq_lte = np.zeros((NO_OF_LAYERS + 1, MAX_NODES_IN_LAYER + 1, MAX_NODES_IN_LAYER + 1))
        ineq_gte = np.zeros((NO_OF_LAYERS + 1, MAX_NODES_IN_LAYER + 1, MAX_NODES_IN_LAYER + 1))
        ineq_relu_gte = np.zeros((NO_OF_LAYERS + 1, MAX_NODES_IN_LAYER + 1, MAX_NODES_IN_LAYER + 1))
        ineq_relu_lte = np.zeros((NO_OF_LAYERS + 1, MAX_NODES_IN_LAYER + 1, MAX_NODES_IN_LAYER + 1))
        if_activation = np.zeros((NO_OF_LAYERS + 1, MAX_NODES_IN_LAYER + 1))
        # TO-DO: can make the assumption if one node in a layer has relu then all do
        for current in nodes:
            if isinstance(current, Function):
                for (node, eq) in zip(current.stmts[0], current.stmts[1]):
                    coeffs = np.zeros((MAX_NODES_IN_LAYER + 1,))
                    eq = self.texpr_to_dict(eq)
                    for var, val in eq.items():
                        if var != '_':
                            var = int(
                                var[2])  # TO-DO: do something more general than assume format X[N][N] for var name
                            coeffs[var] = val
                        else:
                            coeffs[0] = val
                    ineq_lte[int(str(node)[1]), int(str(node)[2]),
                    :] = coeffs  # TO-DO: do something more general than assume format X[N][N] for var name
            elif isinstance(current, Activation):
                if_activation[int(str(current.stmts)[1]), int(str(current.stmts)[2])] = True
            else:
                '''     What to do here
                for stmt in reversed(current.stmts):
                    state = self.semantics.assume_call_semantics(stmt, state, self.manager)'''
                continue

        # can remove layer 0 as it has no inequations
        for i in range(1, len(ineq_lte)):

            print(f"\t\t LAYER {i} Input Equations")
            for j in range(1, len(ineq_lte[0])):
                pass
                print(f"Node: {j} -> if_activation: {if_activation[i, j]}\n eq: {self.ineq_str(ineq_lte[i, j], i, j, '=', i - 1)} ")

            if (i != 1):
                ineq_lte[i], ineq_gte[i] = self.back_propagate_l1_GPU(ineq_lte[i], ineq_relu_lte[i - 1],
                                                                      ineq_relu_gte[i - 1])
            else:
                ineq_gte[i] = ineq_lte[i].copy()

            print(f"\t\t LAYER {i} Substituted")
            for j in range(1, len(ineq_lte[0])):
                print(f"\tNode {j}")
                print(f" eq LTE L1: {self.ineq_str(ineq_lte[i][j], i, j, '>=', 0)}")
                print(f" eq GTE L1: {self.ineq_str(ineq_gte[i][j], i, j, '<=', 0)}")
                print(
                    f" eq (LB,UB): {self.get_bounds_single(ineq_lte[i], ineq_gte[i], j)}")  # Performing the whole debug-print segment in CPU will be removed later.

            if (if_activation[i,1]==1):  # assuming if first node in a layer has activation then all do
                ineq_relu_lte[i], ineq_relu_gte[i] = self.relu_propagate_l1_GPU2(ineq_lte[i], ineq_gte[i])
                print(f"\t RELU-LAYER {i}")
                for j in range(1, len(ineq_lte[0])):
                    print(f"\tNode {j}")
                    print(f" Relu eq LTE L1: {self.ineq_str(ineq_relu_lte[i][j], i, j, '>=', 0)}")
                    print(f" Relu eq GTE L1: {self.ineq_str(ineq_relu_gte[i][j], i, j, '<=', 0)}")
                    print(f"Relu eq (LB,UB): {self.get_bounds_single(ineq_relu_lte[i], ineq_relu_gte[i], j)}")
                # print stuff
            else:
                ineq_relu_lte[i][j], ineq_relu_gte[i][j] = ineq_lte[i][j], ineq_gte[i][j]
                print(f"\t\t NO RELU ON LAYER {i}")


    def analyze(self, initial, earlystop=True, forced_active=None, forced_inactive=None, outputs=None):
        """Forward analysis extracting abstract activation patterns.

        :param initial: initial state of the analysis
        :return: three sets: all activation nodes, always active nodes, always inactive nodes
        """

        worklist = Queue()
        worklist.put(self.cfg.in_node)
        state = deepcopy(initial)
        activated, deactivated = set(), set()

        # Me add
        nodes = []
        for _, node in self.cfg.nodes.items():
            nodes.append(node)
        #self.network_condense_GPU(nodes)
        #SymbolicGPU().network_condense_GPU(nodes,initial)
        #NeurifyGPU().network_condense_GPU(nodes,initial)
        #symbolic_gpu.network_condense_GPU(nodes, initial,outputs)
        #ProductGPU().network_condense_GPU(nodes, initial,{"Neurify","DeepPoly","Symbolic"})
        #deeppoly_gpu.network_condense_GPU(nodes, initial,forced_active=None, forced_inactive=None, outputs=outputs)
        #product_gpu.network_condense_GPU(nodes, initial, {"Neurify", "DeepPoly", "Symbolic"},forced_active=None, forced_inactive=None, outputs=outputs)
        time_sec = time.time()
        product_gpu.network_condense_GPU(nodes, initial, {"Neurify", "DeepPoly", "Symbolic"},forced_active=None, forced_inactive=None, outputs=outputs)
        time_sec = time.time() - time_sec
        print(f"GPU time: {time_sec}\n\n")
        # till here
        while not worklist.empty():
            current: Node = worklist.get()  # retrieve the current node
            # execute block

            if isinstance(current, Function):
                # print(f"{current.stmts}")
                state = state.affine(current.stmts[0], current.stmts[1])
            elif isinstance(current, Activation):
                if forced_active and current in forced_active:
                    state = state.relu(current.stmts, active=True)
                    activated.add(current)
                elif forced_inactive and current in forced_inactive:
                    state = state.relu(current.stmts, inactive=True)
                    deactivated.add(current)
                else:
                    state = state.relu(current.stmts)
                    if state.is_bottom():
                        deactivated.add(current)
                    if state.flag:
                        if state.flag > 0:
                            activated.add(current)
                        else:
                            deactivated.add(current)
            else:
                for stmt in reversed(current.stmts):
                    state = self.semantics.assume_call_semantics(stmt, state, self.manager)
            # update worklist
            ''' ME remove'''
            for node in self.cfg.successors(current):
                worklist.put(self.cfg.nodes[node.identifier])

        self._state_log(state, outputs)
        found = state.outcome(outputs)
        #print(f"DEBUG--> initial: {initial.bounds.items()}")
        '''print("DEBUG -> activated")
        for act in activated:
            print(f"ident: {act.identifier};stmts: {act.stmts}")
        print("DEBUG -> deactivated")
        for deact in deactivated:
            print(f"ident: {deact.identifier};stmts: {type(deact.stmts)}")'''

        pri = []
        for iout in outputs:
            pri.append(iout.name)
        #print(f"DEBUG IN -> forced_active:{forced_active} forced_inactive:{forced_inactive} output:{pri}")
        print(f"DEBUG CPU -> \tinitial:{initial.bounds.items()}\n active:{activated}; deactive:{deactivated}; outcome:{found}")
        return activated, deactivated, found

    def analyze_GPU(self,initial,outputs):
        nodes = []
        for _, node in self.cfg.nodes.items():
            nodes.append(node)
        #activated,deactivated,outcome = neurify_gpu.network_condense_GPU(nodes, initial,outputs)
        #activated, deactivated, outcome = symbolic_gpu.network_condense_GPU(nodes, initial,outputs)
        activated, deactivated, outcome = product_gpu.network_condense_GPU(nodes, initial,{"Neurify","DeepPoly"},outputs)
        # self.network_condense_GPU(nodes)
        '''print("DEBUG -> activated")
        for act in activated:
            print(f"ident: {act.identifier};stmts: {act.stmts}")
        print("DEBUG -> deactivated")
        for deact in deactivated:
            print(f"ident: {deact.identifier};stmts: {type(deact.stmts)}")'''
        print(f"DEBUG GPU -> initial:{initial.bounds.items()};\n active:{activated}; deactive:{deactivated}; outcome:{outcome}")

        return activated,deactivated,outcome


class ActivationPatternForwardSemantics(DefaultForwardSemantics):

    def assume_call_semantics(self, stmt: Call, state: State, manager: PyManager = None) -> State:
        argument = self.semantics(stmt.arguments[0], state).result
        state.assume(argument, manager=manager)
        state.result = set()
        return state
