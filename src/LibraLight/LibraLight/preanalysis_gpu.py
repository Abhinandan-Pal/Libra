import time
import config
from optimized import deeppoly_gpu as dpG
from optimized import symbolic_gpu as symG
from optimized import neurify_gpu as neuG
from optimized import product_gpu as prodG
from optimized import commons as comG

unbiased = []
feasible = dict()
unfeasible = []
unbiasedP = 0
feasibleP = 0
unfeasibleP = 0

def preanalysis(config,L_start,L_min,U):
    print(f"L_start:{L_start}; L_min:{L_min}; U:{U}")
    time_sec = time.time()
    netGPU = comG.createNetworkGPU(config.layers,config.bounds,config.activations,config.sensitive,config.outputs)
    iterPreanalysis(None,None,netGPU,L_start/2,U,L_min,config.sensitive,100.0)
    time_sec = time.time() - time_sec
    print(f"GPU time: {time_sec}\n\n")
    print(f"Unbiased % = {unbiasedP}; feasible % = {feasibleP}; Unfeasible % = {unfeasibleP}")
    print(f"Unbiased: {unbiased}")
    print(f"Unfeasible: {unfeasible}")
    print(f"feasible: {feasible}")
    prioritized = priorityFeasible(compressFeasible(feasible))
    print(f"Prioritized: {prioritized}")
    return prioritized,time_sec

def convertBound(lbL,ubL,inv_var_index,sensitive):
    bound = dict()
    for i in range(1,len(lbL)):
        var = inv_var_index[(0,i)]
        if(str(var) != str(sensitive)):
            bound[var] = (lbL[i],ubL[i])
    return frozenset(bound.items())



def iterPreanalysis(l1_lbL,l1_ubL,netGPU,L,U,L_min,sensitive,percent):
    global unbiasedP,feasibleP,unfeasibleP
    activatedL, deactivatedL, outcomeL, lbL,ubL,percent,inv_var_index = dpG.analyze(netGPU,l1_lbL,l1_ubL,percent)
    l1_lbN,l1_ubN = [],[]
    for activatedL1, deactivatedL1, outcomeL1, lbL1, ubL1 in zip(activatedL, deactivatedL, outcomeL, lbL, ubL):
        for activated, deactivated, outcome, lb, ub in zip(activatedL1, deactivatedL1, outcomeL1, lbL1, ubL1):
            print(f"0:{percent} ; 1:{activated} ; 2:{deactivated} ; 3:{outcome} ; 4:{lb} ; 5:{ub}")
            if(outcome != None):        #unbiased
                unbiased.append((percent,(lb,ub)))
                unbiasedP += percent
                continue
            unknown = len(config.activations) - len(activated) - len(deactivated)
            if(unknown<=U):
                activated = frozenset(activated)
                deactivated = frozenset(deactivated)
                feasibleP += percent
                if ((activated, deactivated) in feasible.keys()):
                    feasible[(activated, deactivated)].add((percent,convertBound(lb,ub,inv_var_index,sensitive)))
                else:
                    feasible[(activated, deactivated)] = set([(percent,convertBound(lb,ub,inv_var_index,sensitive))])
            elif(L/2 >= L_min):
                l1_lbN.append(lb)
                l1_ubN.append(ub)
            else:
                unfeasible.append((lb,ub))
                unfeasibleP += percent
    if(len(l1_lbN) != 0):
        iterPreanalysis(l1_lbN,l1_ubN,netGPU,L/2,U,L_min,sensitive,percent)
    

def compressRange(rangeL):
    def sortKey(val):
        return val[0]
    rangeLN = []
    rangeL.sort(key = sortKey)
    rangeC = rangeL[0]
    for i in range(len(rangeL)-1):
        if(rangeL[i][1] == rangeL[i+1][0]):
            rangeC = (rangeC[0],rangeL[i+1][1])
        else:
            rangeLN.append(rangeC)
            rangeC = rangeL[i+1]
    rangeLN.append(rangeC)
    return rangeLN

def compressFeasible(patterns):
    compressed = dict()
    for key1, pack1 in sorted(patterns.items(), key=lambda v: len(v[1]), reverse=False):
        unmerged = True
        for key2 in compressed:
            (s11, s12) = key1
            (s21, s22) = key2
            if s21.issubset(s11) and s22.issubset(s12):
                unmerged = False
                compressed[key2] = compressed[key2].union(pack1)
                break
            if s11.issubset(s21) and s12.issubset(s22):
                unmerged = False
                compressed[key1] = compressed[key2].union(pack1)
                del compressed[key2]
                break
        if unmerged:
            compressed[key1] = pack1
    return compressed

def priorityFeasible(compressed):
    def max_disj(key):
        return len(config.activations) - len(key[0]) - len(key[1])
    prioritized = sorted(compressed.items(), key=lambda v: max_disj(v[0]) + len(v[1]), reverse=True)
    return prioritized