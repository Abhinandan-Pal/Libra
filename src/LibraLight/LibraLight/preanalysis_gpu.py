import time
import config
from optimized import deeppoly_gpu as dpG
from optimized import symbolic_gpu as symG
from optimized import neurify_gpu as neuG
from optimized import product_gpu as prodG
from optimized import commons as comG
import json

feasible = dict()
unbiasedP = 0
feasibleP = 0
unfeasibleP = 0

def preanalysis(json_out,config,L_start,L_min,U):
    print(f"L_start:{L_start}; L_min:{L_min}; U:{U}")
    json_out["fair"] = []
    json_out["unknown"] = []
    time_sec = time.time()
    netGPU = comG.createNetworkGPU(config.layers,config.bounds,config.activations,config.sensitive,config.outputs)
    iterPreanalysis(json_out,None,None,netGPU,L_start/2,U,L_min,config.sensitive,100.0)
    time_sec = time.time() - time_sec
    print(f"GPU time: {time_sec}\n\n")
    print(f"fair % = {unbiasedP}; feasible % = {unbiasedP+feasibleP}; Unfeasible % = {unfeasibleP}")
    #print(f"Unbiased: {unbiased}")
    #print(f"Unfeasible: {unfeasible}")
    print(f"feasible: {feasible}")
    prioritized = priorityFeasible(compressFeasible(feasible))
    #print(f"Prioritized: {prioritized}")
    return json_out,prioritized,time_sec

def convertBound(lbL,ubL,inv_var_index,sensitive):
    bound = dict()
    for i in range(1,len(lbL)):
        var = inv_var_index[(0,i)]
        if(str(var) != str(sensitive)):
            bound[var] = (lbL[i],ubL[i])
    return frozenset(bound.items())

def boundDict(lbL,ubL,inv_var_index,sensitive):
    jsonDict = dict()
    bound = dict()
    for i in range(1, len(lbL)):
        var = inv_var_index[(0, i)]
        if (str(var) != str(sensitive)):
            temp = dict()
            temp["lower bound"] = lbL[i]
            temp["upper bound"] = ubL[i]
            jsonDict[var] = temp
            bound[var] = (lbL[i], ubL[i])
    return jsonDict,frozenset(bound.items())


def iterPreanalysis(json_out,l1_lbL,l1_ubL,netGPU,L,U,L_min,sensitive,percent):
    global unbiasedP,feasibleP,unfeasibleP
    print(f"PER:{percent}")
    activatedL, deactivatedL, outcomeL, lbL,ubL,percent,inv_var_index = dpG.analyze(netGPU,l1_lbL,l1_ubL,percent)
    l1_lbN,l1_ubN = [],[]
    for activatedL1, deactivatedL1, outcomeL1, lbL1, ubL1 in zip(activatedL, deactivatedL, outcomeL, lbL, ubL):
        for activated, deactivated, outcome, lb, ub in zip(activatedL1, deactivatedL1, outcomeL1, lbL1, ubL1):
            unknown = len(config.activations) - len(activated) - len(deactivated)
            print(f"0:{percent} ; 1:{unknown} ; 3:{outcome} ; 4:{lb} ; 5:{ub} ; L:{L}")
            if(outcome != None):        #unbiased
                jsonBound, _ = boundDict(lb, ub, inv_var_index, sensitive)
                curr = json_out["fair"]
                curr.append(jsonBound)
                json_out["fair"] = curr
                unbiasedP += percent
                continue
            unknown = len(config.activations) - len(activated) - len(deactivated)
            if(unknown<=U):
                activated = frozenset(activated)
                deactivated = frozenset(deactivated)
                feasibleP += percent
                jsonBound, libraBound = boundDict(lb, ub, inv_var_index, sensitive)
                if ((activated, deactivated) in feasible.keys()):
                    feasible[(activated, deactivated)].add((percent,convertBound(lb,ub,inv_var_index,sensitive)))
                else:
                    feasible[(activated, deactivated)] = set([(percent,libraBound)])
            elif(L/2 >= L_min):
                l1_lbN.append(lb)
                l1_ubN.append(ub)
            else:
                jsonBound, _ = boundDict(lb, ub, inv_var_index, sensitive)
                curr = json_out["unknown"]
                curr.append(jsonBound)
                json_out["unknown"] = curr
                unfeasibleP += percent
    if(len(l1_lbN) != 0):
        iterPreanalysis(json_out,l1_lbN,l1_ubN,netGPU,L/2,U,L_min,sensitive,percent)
    

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