import math
from itertools import product
import numpy as np
from apronpy.texpr0 import TexprRtype, TexprRdir, TexprDiscr, TexprOp
from libra.core.cfg import Node, Function, Activation,Basic
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

def ineq_str(ineq,layer_lhs,node_num,op,layer_rhs,inv_var_index):
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
    for i in range(1,len(l1_lb)):
        if (l1_lte[i] < 0):
            lb += l1_lte[i] * l1_ub[i]
        else:
            lb += l1_lte[i] * l1_lb[i]
    ub = l1_gte[0]
    for i in range(1,len(l1_ub)):
        if (l1_gte[i] > 0):
            ub += l1_gte[i] * l1_ub[i]
        else:
            ub += l1_gte[i] * l1_lb[i]
    return lb, ub

def get_bounds_single_neurify(l1_layer_lte,l1_layer_gte,node_num:int , l1_lb,l1_ub,relu_val=[1,0,1,0]):
    l1_lte = l1_layer_lte[node_num]*relu_val[0]

    l1_lte[0] += relu_val[1]

    l1_gte = l1_layer_gte[node_num]*relu_val[2]
    l1_gte[0] += relu_val[3]
    lb = l1_lte[0]
    for i in range(1,len(l1_lte)):
        if (l1_lte[i] < 0):
            lb += l1_lte[i] * l1_ub[i][0]
        else:
            lb += l1_lte[i] * l1_lb[i][0]
    ub = l1_gte[0]
    for i in range(1,len(l1_gte)):
        if (l1_gte[i] > 0):
            ub += l1_gte[i] * l1_ub[i][1]
        else:
            ub += l1_gte[i] * l1_lb[i][1]
    return lb, ub

def getNetShape(nodes):
    NO_OF_LAYERS = 1
    MAX_NODES_IN_LAYER = 1
    CURR_NODED_IN_LAYER = 0
    flag = False
    for current in nodes:
        if isinstance(current, Function):
            for (node, eq) in zip(current.stmts[0], current.stmts[1]):              #TODO just use the length
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

def fillInitials(initial,L,MNIL):
    bounds = []
    NO_OF_INITIALS = 0
    for var, bound in initial.bounds.items():
        gaps = []
        rangeU = bound.upper - bound.lower
        for i in range(math.ceil(rangeU/L)+1):
            gaps.append(bound.lower + L*i)
        bounds.append(product(gaps,gaps))
        NO_OF_INITIALS += 1
    bounds = product(*bounds)
    #for b in bounds:
    #   print(f"Bounds: {b}")
    l1_lb_a,l1_ub_a,l1_lb,l1_ub = [],[],[],[]
    count = 0;
    for bound in bounds:
        l1_lb_t = np.zeros((len(initial.bounds.items()) + 1,))
        l1_ub_t = np.zeros((len(initial.bounds.items()) + 1,))
        flag = False
        for i in range(1,len(initial.bounds.items())+1):
            l1_lb_t[i] = bound[i-1][0]
            l1_ub_t[i] = bound[i-1][1]
            if(l1_lb_t[i]>=l1_ub_t[i]):
                flag = True
                break
        if(flag):
            continue
        l1_lb_a.append(l1_lb_t)
        l1_ub_a.append(l1_ub_t)
        count += 1
        if(count == (2**16)):
            l1_lb_a,l1_ub_a = np.array(l1_lb_a),np.array(l1_ub_a)
            l1_lb.append(l1_lb_a)
            l1_ub.append(l1_ub_a)
            count = 0
            l1_lb_a,l1_ub_a = [],[]
    l1_lb_a,l1_ub_a = np.array(l1_lb_a),np.array(l1_ub_a)
    l1_lb.append(l1_lb_a)
    l1_ub.append(l1_ub_a)
    #print(f"lbs Shape->{l1_lb} ubs Shape->{l1_ub}")
    #for i in range(len(l1_lb[0])):
        #print(f"lbs-> {l1_lb[0][i]}; ubs-> {l1_ub[0][i]}")
    return l1_lb,l1_ub