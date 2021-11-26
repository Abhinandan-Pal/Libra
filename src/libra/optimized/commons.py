import numpy as np
from apronpy.texpr0 import TexprRtype, TexprRdir, TexprDiscr, TexprOp
from apronpy.texpr1 import PyTexpr1
from apronpy.var import PyVar
from copy import deepcopy

def texpr_to_dict( texpr):
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

def ineq_str(ineq:list[float],layer_lhs,node_num,op,layer_rhs):
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

'''Given a node of a layer expressed in inequality form of x01 and x02. it gives the upper and lower bound
 assuming x0n are in [0,1]'''
def get_bounds_single(l1_layer_lte: list[list[float]],l1_layer_gte: list[list[float]],node_num:int , l1_lb = -1,l1_ub = 1):
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