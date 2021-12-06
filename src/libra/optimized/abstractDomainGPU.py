import numpy as np
import cupy as cp
from numba import cuda
from apronpy.texpr0 import TexprRtype, TexprRdir, TexprDiscr, TexprOp
from apronpy.texpr1 import PyTexpr1
from apronpy.var import PyVar
from copy import deepcopy
from libra.core.cfg import Node, Function, Activation,Basic
from libra.core.expressions import VariableIdentifier

class AbstractDomainGPU():
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
                    lbs[id] += l1_lte[id, i] * l1_ub[i]
                else:
                    lbs[id] += l1_lte[id, i] * l1_lb[i]
            ubs[id] = l1_gte[id, 0]
            for i in range(1, len(l1_gte[id])):
                if (l1_gte[id, i] > 0):
                    ubs[id] += l1_gte[id, i] * l1_ub[i]
                else:
                    ubs[id] += l1_gte[id, i] * l1_lb[i]

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

    def texpr_to_dict(self,texpr):
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
                    result = dict()
                    if '_' in left and len(left) == 1:
                        for var, val in right.items():
                            result[var] = left['_'] * right[var]
                    elif '_' in right and len(right) == 1:
                        for var, val in left.items():
                            result[var] = right['_'] * left[var]
                    else:
                        assert False
                elif op == TexprOp.AP_TEXPR_NEG:
                    result = deepcopy(left)
                    for var, val in result.items():
                        result[var] = -val
            return result

        texpr1 = texpr.texpr1.contents
        return do(texpr1.texpr0.contents, texpr1.env.contents)

    def ineq_str(self,ineq, layer_lhs, node_num, op, layer_rhs, inv_var_index):
        if not ((layer_lhs, node_num) in inv_var_index.keys()):
            return "Empty Node"
        if (op == ">="):
            str_ineq = f"{inv_var_index[(layer_lhs, node_num)]} >= "
        elif (op == "="):
            str_ineq = f"{inv_var_index[(layer_lhs, node_num)]} = "
        else:
            str_ineq = f"{inv_var_index[(layer_lhs, node_num)]} <= "
        for i in range(len(ineq)):
            if (i == 0):
                str_ineq += f"+({ineq[0]})"
            else:
                if ((layer_rhs, i) in inv_var_index.keys()):
                    str_ineq += f"+({ineq[i]} * {inv_var_index[(layer_rhs, i)]})"
        return str_ineq

    def ineq_str_direct(self,ineq, layer_lhs, node_num, op, layer_rhs):
        if (op == ">="):
            str_ineq = f"X{layer_lhs}{node_num} >= "
        elif (op == "="):
            str_ineq = f"X{layer_lhs}{node_num} = "
        else:
            str_ineq = f"X{layer_lhs}{node_num} <= "
        for i in range(len(ineq)):
            if (i == 0):
                str_ineq += f"+({ineq[0]})"
            else:
                str_ineq += f"+({ineq[i]} * X{layer_rhs}{i})"
        return str_ineq

    '''Given a node of a layer expressed in inequality form of x01 and x02. it gives the upper and lower bound
     assuming x0n are in [0,1]'''

    def get_bounds_single(self,l1_layer_lte, l1_layer_gte, node_num: int, l1_lb, l1_ub, relu_val=[1, 0, 1, 0]):
        l1_lte = l1_layer_lte[node_num] * relu_val[0]

        l1_lte[0] += relu_val[1]

        l1_gte = l1_layer_gte[node_num] * relu_val[2]
        l1_gte[0] += relu_val[3]
        lb = l1_lte[0]
        for i in range(1, len(l1_lte)):
            if (l1_lte[i] < 0):
                lb += l1_lte[i] * l1_ub[i]
            else:
                lb += l1_lte[i] * l1_lb[i]
        ub = l1_gte[0]
        for i in range(1, len(l1_gte)):
            if (l1_gte[i] > 0):
                ub += l1_gte[i] * l1_ub[i]
            else:
                ub += l1_gte[i] * l1_lb[i]
        return lb, ub

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
        return NO_OF_LAYERS,MAX_NODES_IN_LAYER

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