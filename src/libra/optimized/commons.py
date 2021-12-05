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

def ineq_str(ineq:list[float],layer_lhs,node_num,op,layer_rhs,inv_var_index):
    if not ((layer_lhs,node_num) in inv_var_index.keys()):
        return "Empty Node"
    if(op==">="):
        str_ineq = f"{inv_var_index[(layer_lhs,node_num)]} >= "
    elif (op == "="):
        str_ineq = f"{inv_var_index[(layer_lhs,node_num)]} = "
    else:
        str_ineq = f"{inv_var_index[(layer_lhs,node_num)]} <= "
    for i in range(len(ineq)):
        if(i==0):
            str_ineq += f"+({ineq[0]})"
        else:
            if((layer_rhs,i) in inv_var_index.keys()):
                str_ineq += f"+({ineq[i]} * {inv_var_index[(layer_rhs,i)]})"
    return str_ineq

def ineq_str_direct(ineq:list[float],layer_lhs,node_num,op,layer_rhs):
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
def get_bounds_single(l1_layer_lte,l1_layer_gte,node_num:int , l1_lb,l1_ub,relu_val=[1,0,1,0]):
    l1_lte = l1_layer_lte[node_num]*relu_val[0]

    l1_lte[0] += relu_val[1]

    l1_gte = l1_layer_gte[node_num]*relu_val[2]
    l1_gte[0] += relu_val[3]
    lb = l1_lte[0]
    for i in range(1,len(l1_lte)):
        if (l1_lte[i] < 0):
            lb += l1_lte[i] * l1_ub[i]
        else:
            lb += l1_lte[i] * l1_lb[i]
    ub = l1_gte[0]
    for i in range(1,len(l1_gte)):
        if (l1_gte[i] > 0):
            ub += l1_gte[i] * l1_ub[i]
        else:
            ub += l1_gte[i] * l1_lb[i]
    return lb, ub