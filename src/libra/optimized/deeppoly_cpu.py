import numpy as np
#replace with numpy implementaions
from libra.optimized.commons import texpr_to_dict,get_bounds_single,ineq_str
from libra.core.cfg import Node, Function, Activation
import copy

'''lte stores equation < Xn gte store equation > Xn'''
'''If Previous layer is expressed in inequality form of x01 and x02  back_propagate_l1 expresses a node of 
current layer in inequality form of x01 and x02 '''
def back_propagate_l1( eq_layer: list[list[float]], ineq_prev_lte: list[list[float]],
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

'''If current layer is expressed in inequality form of x01 and x02  relu_propagate_l1 expresses a node of
	current layer after relu in inequality form of x01 and x02 '''


def relu_propagate_l1(l1_layer_lte: list[list[float]], l1_layer_gte: list[list[float]], node_num: int):
	l1_lte = l1_layer_lte[node_num]
	l1_gte = l1_layer_gte[node_num]
	lb, ub = get_bounds_single(l1_layer_lte, l1_layer_gte, node_num)
	l1_relu_lte = [0] * len(l1_lte)
	l1_relu_gte = [0] * len(l1_gte)
	'''Case 1(Strictly Negative)'''
	if (ub < 0):
		return l1_relu_lte, l1_relu_gte
	'''Case 2(Strictly Positive)'''
	if (lb > 0):
		return l1_lte, l1_gte
	'''Case 3(Crossing Relu)'''

	slope = ub / (ub - lb)
	y_coeff = -ub * lb / (ub - lb)
	for i in range(len(l1_gte)):
		l1_relu_gte[i] = slope * l1_gte[i]
	l1_relu_gte[0] += y_coeff
	b3_area = abs(ub * (ub - lb))
	c3_area = abs(lb * (ub - lb))
	print(f"DEBUG ---> b3_area = {b3_area}, c3_area = {c3_area}")
	if (c3_area < b3_area):
		for i in range(len(l1_lte)):
			l1_relu_lte[i] = l1_lte[i]
	return l1_relu_lte, l1_relu_gte

def network_condense_CPU( nodes):
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
				eq = texpr_to_dict(eq)
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
			state = semantics.assume_call_semantics(stmt, state, manager)'''
	print(f'{ineq_lte}')
	for i in range(1, len(ineq_lte)):
		for j in range(1, len(ineq_lte[0])):
			print(f"\t\tX{i}{j}\n if_activation: {if_activation[i][j]}\n eq: {ineq_str(ineq_lte[i][j],i,j,'=',i-1)} ")
			if(i != 1):
				ineq_lte[i][j],ineq_gte[i][j] = back_propagate_l1(ineq_lte[i],ineq_relu_lte[i-1],ineq_relu_gte[i-1], j)
			else:
				ineq_gte[i][j] = copy.deepcopy(ineq_lte[i][j])
			print(f" eq LTE L1: {ineq_str(ineq_lte[i][j],i,j,'>=',0)}")                                #Use deep copy or on relu values are changing
			print(f" eq GTE L1: {ineq_str(ineq_gte[i][j], i, j, '<=', 0)}")
			print(f"eq (LB,UB): {get_bounds_single(ineq_lte[i],ineq_gte[i],j)}")

			if(if_activation[i][j]):
				ineq_relu_lte[i][j], ineq_relu_gte[i][j] = relu_propagate_l1(ineq_lte[i],ineq_gte[i],j)
			else:
				ineq_relu_lte[i][j], ineq_relu_gte[i][j] = ineq_lte[i][j], ineq_gte[i][j]

			print(f" Relu eq LTE L1: {ineq_str(ineq_relu_lte[i][j], i, j, '>=', 0)}")
			print(f" Relu eq GTE L1: {ineq_str(ineq_relu_gte[i][j], i, j, '<=', 0)}")
			print(f"Relu eq (LB,UB): {get_bounds_single(ineq_relu_lte[i], ineq_relu_gte[i], j)}")
