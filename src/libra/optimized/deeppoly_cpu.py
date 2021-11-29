import numpy as np
#replace with numpy implementaions
from libra.optimized.commons import texpr_to_dict,get_bounds_single,ineq_str
from libra.core.cfg import Node, Function, Activation
import copy

def back_propagate(affine,relu,layer:int,if_activation):
	ln_coeff_lte = affine[layer].copy()
	ln_coeff_gte = affine[layer].copy()

	def back_affine(ineq_prev_lte,ineq_prev_gte,ln_coeff_lte,ln_coeff_gte):
		l1_lte = np.zeros(affine[layer].shape)
		l1_gte = np.zeros(affine[layer].shape)
		l1_lte[:,0] = ln_coeff_lte[:,0]
		l1_gte[:,0] = ln_coeff_gte[:,0]
		for i in range(1, len(l1_lte)):		#loop through each coeff of a node of current layer and nodes of previous layer
			for k in range(0,len(l1_lte)):	#loop through all nodes
				for p in range(0, len(ineq_prev_lte[1])):	#loop thorugh coeffients of a node of previous layer.
					if(ln_coeff_lte[k][i]>0):
						l1_lte[k][p] += ln_coeff_lte[k][i] * ineq_prev_lte[i][p]            #should it be i or i-1?
					else:
						l1_lte[k][p] += ln_coeff_lte[k][i] * ineq_prev_gte[i][p]
					if(ln_coeff_gte[k][i]>0):
						l1_gte[k][p] += ln_coeff_gte[k][i] * ineq_prev_gte[i][p]
					else:
						l1_gte[k][p] += ln_coeff_gte[k][i] * ineq_prev_lte[i][p]
		return l1_lte,l1_gte
	def back_relu(relu_layer,ln_coeff_lte,ln_coeff_gte):
		for i in range(1,len(ln_coeff_lte)):
			for j in range(1,len(relu_layer)):
				if(ln_coeff_lte[i][j]>0):
					ln_coeff_lte[i][0] += relu_layer[i][1] * ln_coeff_lte[i][j]
					ln_coeff_lte[i][j] = relu_layer[i][0]*ln_coeff_lte[i][j]			#[1:] to make sure base term is not affected
				else:
					ln_coeff_lte[i][0] += relu_layer[i][3] * ln_coeff_lte[i][j]
					ln_coeff_lte[i][j] = relu_layer[i][2] * ln_coeff_lte[i][j]
				if(ln_coeff_gte[i][j]>0):
					ln_coeff_gte[i][0] += relu_layer[i][3] * ln_coeff_gte[i][j]
					ln_coeff_gte[i][j] = relu_layer[i][2] * ln_coeff_gte[i][j]
				else:
					ln_coeff_gte[i][0] += relu_layer[i][1] * ln_coeff_gte[i][j]
					ln_coeff_gte[i][j] = relu_layer[i][0] * ln_coeff_gte[i][j]

	layer_t = layer
	while(layer!= 1):	#layer zero is input and layer one is in already in terms of input
		#First relu of previous layer
		if(if_activation[layer-1][1]==True):
			#print(f"DEBUG--->relu:{relu[layer-1]}; ln_gte={ln_coeff_gte}")
			back_relu(relu[layer-1],ln_coeff_lte,ln_coeff_gte)
			#print(f"DEBUG---> ln_lte AFTER ={ln_coeff_gte}")

		#Then affine of previous layer
		ineq_prev_gte = affine[layer-1]
		ineq_prev_lte = affine[layer-1]
		ln_coeff_lte,ln_coeff_gte = back_affine(ineq_prev_lte,ineq_prev_gte,ln_coeff_lte,ln_coeff_gte)
		layer -= 1
	if(if_activation[layer_t][1]==1):
		#print("DEBUG --> PERFORM RELU")
		relu[layer_t],active_pattern = relu_propagate_l1_CPU2(ln_coeff_lte,ln_coeff_gte)
		print(f"DEBUG RELU layer: {layer_t} ---> {relu[layer_t]}")
	else:
		pass
		#print("DEBUG --> DONT PERFORM RELU")
	#return active_pattern

	#Different return for debug purposes
	return ln_coeff_lte,ln_coeff_gte


def get_bounds_CPU2(l1_lte,l1_gte , l1_lb = -1,l1_ub = 1):
	lbs = np.zeros(l1_lte.shape[0])
	ubs = np.zeros(l1_lte.shape[0])
	for i in range(len(l1_lte)):
		lbs[i] = l1_lte[i][0]
		for coeff in l1_lte[i][1:]:
			if (coeff < 0):
				lbs[i] += coeff * l1_ub
			else:
				lbs[i] += coeff* l1_lb
		ubs[i] = l1_gte[i][0]
		for coeff in l1_gte[i][1:]:
			if (coeff > 0):
				ubs[i] += coeff * l1_ub
			else:
				ubs[i] += coeff* l1_lb
	return lbs, ubs

def relu_propagate_l1_CPU2(l1_lte,l1_gte):
	lbs,ubs = get_bounds_CPU2(l1_lte,l1_gte)
	relu_layer = np.zeros((len(l1_lte),4))
	active_pattern = np.zeros(l1_lte.shape[0])
	for i in range(len(l1_lte)):
		if(ubs[i] < 0):
			relu_layer[i] = [0,0,0,0]
			active_pattern[i] = 0
			continue
		elif(lbs[i] > 0):
			relu_layer[i] = [1,0,1,0]
			active_pattern[i] = 1
		else:
			active_pattern[i] = 2
			#print(f"DEBUG ----> ubs: {ubs[i]};lbs {lbs[i]}")
			slope = ubs[i]/(ubs[i]-lbs[i])
			y_coeff = -ubs[i]*lbs[i]/(ubs[i]-lbs[i])
			#print(f"DEBUG ----> slope: {slope};y_coeff {y_coeff}")
			relu_layer[i] = [0, 0, slope, y_coeff]
			b3_area = abs(ubs[i]*(ubs[i]-lbs[i]))
			c3_area = abs(lbs[i]*(ubs[i]-lbs[i]))
			if(c3_area < b3_area):
				relu_layer[i] = [1, 0, slope, y_coeff]
	return relu_layer,active_pattern


def network_condense_CPU(nodes):
	# equation[n1][n2] stores the bias and coeff of nodes of previous layer to form x[n1][n2] in order
	# if_activation[n1][n2] stores if there is an activation on x[n1][n2] (only relu considered for now)
	NO_OF_LAYERS = 3
	MAX_NODES_IN_LAYER = 2
	affine = np.zeros((NO_OF_LAYERS + 1, MAX_NODES_IN_LAYER + 1, MAX_NODES_IN_LAYER + 1))
	relu = np.zeros((NO_OF_LAYERS + 1, MAX_NODES_IN_LAYER + 1, 4))
	if_activation = np.zeros((NO_OF_LAYERS + 1, MAX_NODES_IN_LAYER + 1))
	# The 4 in relu is for lessThan(slope,y-coeff);greaterThan(slope,y-coeff)
	# TO-DO: can make the assumption if one node in a layer has relu then all do
	for current in nodes:
		if isinstance(current, Function):
			for (node, eq) in zip(current.stmts[0], current.stmts[1]):
				coeffs = np.zeros((MAX_NODES_IN_LAYER + 1,))
				eq = texpr_to_dict(eq)
				for var, val in eq.items():
					if var != '_':
						var = int(
							var[2])  # TO-DO: do something more general than assume format X[N][N] for var name
						coeffs[var] = val
					else:
						coeffs[0] = val
				affine[int(str(node)[1]), int(str(node)[2]),:] = coeffs  # TO-DO: do something more general than assume format X[N][N] for var name
		elif isinstance(current, Activation):
			if_activation[int(str(current.stmts)[1]), int(str(current.stmts)[2])] = True
		else:
			'''What to do here
			for stmt in reversed(current.stmts):
			state = semantics.assume_call_semantics(stmt, state, manager)'''
			continue
	# can remove layer 0 as it has no inequations
	# All these print are for debug mode. Actual will only activation pattern.
	for i in range(1, len(affine)):

		print(f"\t\t LAYER {i} Input Equations")
		for j in range(1, len(affine[0])):
			print(f"Node: {j} -> if_activation: {if_activation[i, j]}\n eq: {ineq_str(affine[i, j], i, j, '=', i - 1)} ")

		ineq_lte, ineq_gte = back_propagate(affine,relu,i,if_activation)

		print(f"\t\t LAYER {i} Substituted")
		for j in range(1, len(affine[0])):
			print(f"\tNode {j}")
			print(f" eq LTE L1: {ineq_str(ineq_lte[j], i, j, '>=', 0)}")
			print(f" eq GTE L1: {ineq_str(ineq_gte[j], i, j, '<=', 0)}")
			print(f" eq (LB,UB): {get_bounds_single(ineq_lte, ineq_gte, j)}")  # Performing the whole debug-print segment in CPU will be removed later.

		if (if_activation[i,1]==1):  # assuming if first node in a layer has activation then all do
			print(f"\t RELU-LAYER {i}")
			for j in range(1, len(affine[0])):
				print(f"\tNode {j}")
				print(f" Relu eq LTE: Slope: {relu[i][j][0]}, Y-Coeff: {relu[i][j][1]}")
				print(f" Relu eq GTE: Slope: {relu[i][j][2]}, Y-Coeff: {relu[i][j][3]}")
				print(f"Relu eq (LB,UB): {get_bounds_single(ineq_lte, ineq_gte, j,relu_val=relu[i][j])}")
			# print stuff
		else:
			print(f"\t\t NO RELU ON LAYER {i}")
