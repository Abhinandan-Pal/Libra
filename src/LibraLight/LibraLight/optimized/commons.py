import math
from itertools import product
import numpy as np
import cupy as cp

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

def convertInitial(bounds,var_index,sensitive):
    row_id, col_id, flag = (0, 1, False)
    max_diff = 0
    l1_lb= np.zeros((1,len(bounds.items()) + 1))
    l1_ub = np.zeros((1,len(bounds.items()) + 1))
    sens = None
    for var, bound in bounds.items():
        if(str(var) == str(sensitive)):
            sens = col_id
        var_index[str(var)] = (row_id, col_id)
        l1_lb[0][col_id] = bound[0]
        l1_ub[0][col_id] = bound[1]
        if(bound[1]-bound[0]>max_diff):
            max_diff = bound[1]-bound[0]
        col_id += 1
    return l1_lb,l1_ub,sens,max_diff

'''
def splitInitial2(l1_lbL,l1_ubL,sensitive): #remove
    l1_lb_a, l1_ub_a, l1_lbLN, l1_ubLN = [], [], [], []
    count = 0
    for (l1_lb, l1_ub) in zip(l1_lbL, l1_ubL):

        for index in range(len(l1_lb)):
            #print(f"c = {count}")
            if(index == sensitive) or math.isclose(l1_lb[index], l1_ub[index]):
                continue
            lb1,ub1,lb2,ub2 = l1_lb.copy(), l1_ub.copy(),l1_lb.copy(),l1_ub.copy()
            mid = l1_lb[index] + (l1_ub[index]-l1_lb[index])/2
            ub1[index] = mid
            lb2[index] = mid
            l1_lb_a.append(lb1)
            l1_lb_a.append(lb2)
            l1_ub_a.append(ub1)
            l1_ub_a.append(ub2)
            count += 2
            if (count == (2 ** 16)):
                l1_lb_a, l1_ub_a = np.array(l1_lb_a), np.array(l1_ub_a)
                l1_lbLN.append(l1_lb_a)
                l1_ubLN.append(l1_ub_a)
                l1_lb_a, l1_ub_a = [], []
                count = 0
    l1_lb_a, l1_ub_a = np.array(l1_lb_a), np.array(l1_ub_a)
    l1_lbLN.append(l1_lb_a)
    l1_ubLN.append(l1_ub_a)
    return l1_lbLN, l1_ubLN
'''

def splitInitial(l1_lbL,l1_ubL,sensitive,L_min):
    bounds = []
    for (l1_lb, l1_ub) in zip(l1_lbL, l1_ubL):
        bnd = []
        for index in range(len(l1_lb)):
            gaps = []
            if (index == sensitive):
                gaps.append((l1_lb[index], l1_ub[index]))
            elif math.isclose(l1_lb[index], l1_ub[index]):
                gaps.append((l1_lb[index], l1_lb[index]))
            elif (l1_ub[index] - l1_lb[index] <= L_min):
                #raise NotImplementedError       #implemented but not tested
                gaps.append((l1_lb[index], l1_ub[index]))
            else:
                mid = l1_lb[index] + (l1_ub[index]-l1_lb[index])/2
                gaps.append((l1_lb[index], mid))
                gaps.append((mid, l1_ub[index]))
            bnd.append(gaps)
        bnd = product(*bnd)
        bounds.extend(bnd)
    #for b in bounds:
    #   print(f"Bounds lb: {b}")
    l1_lb_a, l1_ub_a, l1_lb, l1_ub = [], [], [], []
    count = 0;
    for bound in bounds:
        #print(f"{count}")
        l1_lb_t = np.zeros((len(l1_lbL[0]) ,))
        l1_ub_t = np.zeros((len(l1_lbL[0]),))
        for i in range(len(l1_lbL[0])):
            l1_lb_t[i] = bound[i][0]
            l1_ub_t[i] = bound[i][1]
        l1_lb_a.append(l1_lb_t)
        l1_ub_a.append(l1_ub_t)
        count += 1
        if (count == (2 ** 16)):
            l1_lb_a, l1_ub_a = np.array(l1_lb_a), np.array(l1_ub_a)
            l1_lb.append(l1_lb_a)
            l1_ub.append(l1_ub_a)
            l1_lb_a, l1_ub_a = [], []
            count = 0
    l1_lb_a, l1_ub_a = np.array(l1_lb_a), np.array(l1_ub_a)
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

def createNetworkGPU(layers,bounds,activations,sensitive,outputs):
    NO_OF_LAYERS, MAX_NODES_IN_LAYER = getNetShape(layers)
    print(f"DOMAINS -> DEEPPOLY")
    var_index: dict(str, (int, int)) = dict()
    affine = np.zeros((NO_OF_LAYERS + 1, MAX_NODES_IN_LAYER + 1, MAX_NODES_IN_LAYER + 1)).astype(np.float32)
    if_activation = np.zeros((NO_OF_LAYERS + 1, MAX_NODES_IN_LAYER + 1)).astype(np.int16)
    dims = np.ones(NO_OF_LAYERS + 1).astype(np.int32)

    l1_lb, l1_ub, sensitive,max_diff = convertInitial(bounds,var_index,sensitive)
    fillInput(layers, activations, affine, dims, if_activation, var_index, MAX_NODES_IN_LAYER)
    inv_var_index = {v: k for k, v in var_index.items()}
    outNodes = set()
    for output in outputs:
        outNodes.add(var_index[str(output)][1])
    d_affine = cp.asarray(affine)
    d_if_activation = cp.asarray(if_activation)
    return d_affine,if_activation,d_if_activation,var_index,inv_var_index,outNodes,dims,l1_lb, l1_ub,sensitive,max_diff,NO_OF_LAYERS,MAX_NODES_IN_LAYER
