import math
from itertools import product
import numpy as np

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

def fillInitials(initial,sensative,L):
    bounds = []
    NO_OF_INITIALS = 0
    for var, bound in initial.items():
        gaps = []
        if(str(var) == str(sensative)):
            print(f"SENSATIVE: {var}")
            gaps.append((bound[0], bound[1]))
        elif math.isclose(bound[1],bound[0]):
            gaps.append((bound[0], bound[0]))
        else:
            rangeU = bound[1] - bound[0]
            for i in range(math.ceil(rangeU/L)):
                gaps.append((bound[0] + L*(i),bound[0] + L*(i+1)))
        bounds.append(gaps)
        NO_OF_INITIALS += 1
    bounds = product(*bounds)
    #for b in bounds:
    #   print(f"Bounds lb: {b}")
    l1_lb_a,l1_ub_a,l1_lb,l1_ub = [],[],[],[]
    count = 0;
    for bound in bounds:
        #print(f"{count}")
        l1_lb_t = np.zeros((len(initial.items()) + 1,))
        l1_ub_t = np.zeros((len(initial.items()) + 1,))
        for i in range(1,len(initial)+1):
            #print(f"Bounds lb: {bound[i][0]} ub: {bound[i][1]}")
            l1_lb_t[i] = bound[i-1][0]
            l1_ub_t[i] = bound[i-1][1]
        l1_lb_a.append(l1_lb_t)
        l1_ub_a.append(l1_ub_t)
        count += 1
        if(count == (2**16)):
            l1_lb_a,l1_ub_a = np.array(l1_lb_a),np.array(l1_ub_a)
            l1_lb.append(l1_lb_a)
            l1_ub.append(l1_ub_a)
            l1_lb_a,l1_ub_a = [],[]
            count = 0
    l1_lb_a,l1_ub_a = np.array(l1_lb_a),np.array(l1_ub_a)
    l1_lb.append(l1_lb_a)
    l1_ub.append(l1_ub_a)
    #print(f"lbs Shape->{len(l1_lb)} ubs Shape->{len(l1_ub)}")
    #for i in range(len(l1_lb[0])):
    #    print(f"lbs-> {l1_lb[0][i]}; ubs-> {l1_ub[0][i]}")
    return l1_lb, l1_ub

def getNetShape(layers):
    NO_OF_LAYERS = len(layers)
    MAX_NODES_IN_LAYER = 0
    flag = False
    for layer in layers:
        for coeffs in layer.values():              #TODO just use the length
            MAX_NODES_IN_LAYER = max(MAX_NODES_IN_LAYER, len(coeffs)-1)     #-1 as one of the coeff is from base term
    return NO_OF_LAYERS,MAX_NODES_IN_LAYER

def fillInput(layers,activations,affine,dims,if_activation,var_index,MNIL):
    row_id,col_id = (1,1)
    for layer in layers:
        for lhs, eq in layer.items():
            coeffs = np.zeros((MNIL + 1,)).astype(np.float32)
            for var, val in eq.items():
                if var != '_':
                    r, c = var_index[str(var)]
                    if (r != row_id - 1):
                        raise NotImplementedError(f"Affine should only be based on previous layer {row_id} But was on {r}.")
                    coeffs[c] += val
                else:
                    coeffs[0] += val
            dims[row_id] += 1
            affine[row_id, col_id, :] = coeffs
            var_index[str(lhs)] = (row_id, col_id)
            col_id += 1
        row_id += 1
        col_id = 1
    for var in activations:
        if_activation[var_index[var]] = 1


