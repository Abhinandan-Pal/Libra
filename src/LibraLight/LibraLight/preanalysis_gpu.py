import time
import config
from optimized import deeppoly_gpu as dpG
from optimized import symbolic_gpu as symG
from optimized import neurify_gpu as neuG
from optimized import product_gpu as prodG
from optimized import commons as comG
from colorama import Style, Fore



feasible = dict()
unbiasedP = 0
feasibleP = 0
unfeasibleP = 0
exploredP = 0
unbiased = []
unfeasible = []

def preanalysis(json_out,config,L_start,L_min,U,domains):
    global feasible,unbiasedP,feasibleP,unfeasibleP,unbiased,unfeasible
    feasible,unbiasedP,feasibleP,unfeasibleP,unbiased,unfeasible = dict(),0,0,0,[],[]
    print(f"L_start:{L_start}; L_min:{L_min}; U:{U}")
    json_out["fair"] = []
    json_out["unknown"] = []
    time_sec = time.time()
    netGPU = comG.createNetworkGPU(config.layers,config.bounds,config.activations,config.sensitive,config.outputs)
    iterPreanalysis(None,None,netGPU,L_start/2,U,L_min,config.sensitive,100.0,domains)
    time_sec = time.time() - time_sec

    unbiased = compressRange(unbiased)
    unfeasible = compressRange(unfeasible)
    #feasible = compressRangeFeasible(feasible)

    inv_var_index = netGPU[4]
    feasible = convertFeasible(feasible, inv_var_index, config.sensitive)
    updateJSON(json_out, inv_var_index, config.sensitive, unbiased, unfeasible)
    print(f"fair % = {unbiasedP}; feasible % = {unbiasedP + feasibleP}; Unfeasible % = {unfeasibleP}")
    feasiblePe,fairP = (unbiasedP + feasibleP),unbiasedP
    print(f"GPU time: {time_sec}\n\n")
    prioritized = priorityFeasible(compressFeasible(feasible))
    return json_out,prioritized,time_sec,feasiblePe,fairP

def ppRanges(percent, lb, ub,inv_var_index,sensitive):
    s = "Ranges: "
    for i in range(1,len(lb)):
        var = inv_var_index[(0, i)]
        if (var == sensitive) or (lb[i] == ub[i]):
            continue
        s += f"{var} ∈ [{lb[i]},{ub[i]}]; "
    s += f"| Percent: {percent}"
    return s

def iterPreanalysis(l1_lbL,l1_ubL,netGPU,L,U,L_min,sensitive,percent,domains):
    global unbiasedP,feasibleP,unfeasibleP,exploredP
    activatedL2, deactivatedL2, outcomeL2, lbL2, ubL2, percent, inv_var_index = prodG.analyze(netGPU,l1_lbL,l1_ubL,percent,domains)
    l1_lbN,l1_ubN = [],[]
    for activatedL1, deactivatedL1, outcomeL1, lbL1, ubL1 in zip(activatedL2, deactivatedL2, outcomeL2, lbL2, ubL2):
        for activated, deactivated, outcome, lb, ub in zip(activatedL1, deactivatedL1, outcomeL1, lbL1, ubL1):
            unknown = len(config.activations) - len(activated) - len(deactivated)
            #print(f"0:{percent} ; 1:{unknown} ; 3:{outcome} ; 4:{lb} ; 5:{ub} ; L:{L}")
            ppRan = ppRanges(percent, lb, ub, inv_var_index,sensitive)
            print(Fore.LIGHTMAGENTA_EX +f"{ppRan}", Style.RESET_ALL)
            if(outcome != None):        #unbiased
                unbiased.append([lb,ub,percent,outcome])
                unbiasedP += percent
                exploredP += percent
                print(Fore.GREEN + f"No Bias (Outcome: {outcome}) in {ppRan}",Style.RESET_ALL)
                print(Fore.YELLOW + f"Progress : {unbiasedP+feasibleP}% of {exploredP}% ({unbiasedP}% fair)", Style.RESET_ALL)
                continue
            if(unknown<=U):
                activated = frozenset(activated)
                deactivated = frozenset(deactivated)
                feasibleP += percent
                exploredP += percent
                print(Fore.LIGHTYELLOW_EX + f"Possible Bias in {ppRan}",Style.RESET_ALL)
                print(Fore.YELLOW + f"Progress : {unbiasedP + feasibleP}% of {exploredP}% ({unbiasedP}% fair)",Style.RESET_ALL)
                if ((activated, deactivated) in feasible.keys()):
                    feasible[(activated, deactivated)].append([lb,ub,percent,outcome])
                else:
                    feasible[(activated, deactivated)] = [[lb,ub,percent,outcome]]
            elif(L/2 >= L_min):
                print(f"Too many disjunctions ({unknown})!\nRange can be further split!")
                l1_lbN.append(lb)
                l1_ubN.append(ub)
            else:
                unfeasible.append([lb, ub,percent,None])
                unfeasibleP += percent
                exploredP += percent
                print(f"Too many disjunctions ({unknown})!\nCannot range split anymore!")
                print(Fore.RED + f"Unchecked Bias in {ppRan}", Style.RESET_ALL)
                print(Fore.YELLOW + f"Progress : {unbiasedP + feasibleP}% of {exploredP}% ({unbiasedP}% fair)", Style.RESET_ALL)
    if(len(l1_lbN) != 0):
        iterPreanalysis(l1_lbN,l1_ubN,netGPU,L/2,U,L_min,sensitive,percent,domains)

def boundDict1(lbL,ubL,inv_var_index,sensitive):
    jsonDict = dict()
    for i in range(1, len(lbL)):
        var = inv_var_index[(0, i)]
        if (str(var) != str(sensitive)):
            temp = dict()
            temp["lower bound"] = lbL[i]
            temp["upper bound"] = ubL[i]
            jsonDict[var] = temp
    return jsonDict

def boundDict2(lbL,ubL,inv_var_index,sensitive):
    bound = dict()
    for i in range(1, len(lbL)):
        var = inv_var_index[(0, i)]
        if (str(var) != str(sensitive)):
            bound[var] = (lbL[i], ubL[i])
    return frozenset(bound.items())

def updateJSON(json_out,inv_var_index, sensitive,unbiased,unfeasible):
    def genTemp(bound):
        lb, ub, percent, outcome = bound
        temp = dict()
        temp["ranges"] = boundDict1(lb, ub, inv_var_index, sensitive)
        temp["outcome"] = outcome
        temp["percent"] = percent
        return temp
    for bound in unbiased:
        json_out["fair"].append(genTemp(bound))
    for bound in unfeasible:
        json_out["unknown"].append(genTemp(bound))

def convertBound(lbL,ubL,inv_var_index,sensitive):
    bound = dict()
    for i in range(1,len(lbL)):
        var = inv_var_index[(0,i)]
        if(str(var) != str(sensitive)):
            bound[var] = (lbL[i],ubL[i])
    return frozenset(bound.items())



def convertFeasible(feasibles,inv_var_index,sensitive):
    feasiblesN = dict()
    for key,feasibleL in feasibles.items():
        temp = set()
        for feasible in feasibleL:
            lb, ub, percent, outcome = feasible
            temp.add( (percent,convertBound(lb,ub,inv_var_index,sensitive) ) )
        feasiblesN[key] = temp
    return feasiblesN

def compressRange(rangeL):
    flagOut = True
    if(len(rangeL)==0):
        return rangeL
    while(flagOut):
        flagOut = False
        rangeLN = []
        rangeC = rangeL[0]
        for i in range(1,len(rangeL)):
            pos = -1
            flag = True
            for j in range(len(rangeL[0][0])):
                if(rangeC[3] != rangeL[i][3]):
                    flag = False
                    break
                if(rangeC[1][j] == rangeL[i][0][j]) and (rangeC[0][j] != rangeL[i][1][j]) and (pos == -1):
                    pos = j
                elif(rangeC[1][j] == rangeL[i][0][j]) and (rangeC[0][j] != rangeL[i][1][j]) and (pos != -1):
                    flag = False
                    break
                elif(rangeC[1][j] != rangeL[i][1][j]) or (rangeC[0][j] != rangeL[i][0][j]):
                    flag = False
                    break
            if(flag):
                if(pos != -1):
                    rangeC[1][pos] = rangeL[i][1][pos]
                    rangeC[2] += rangeL[i][2]
                    flagOut = True
            else:
                rangeLN.append(rangeC)
                rangeC = rangeL[i]
        rangeLN.append(rangeC)
        rangeL = rangeLN
    return rangeLN

def compressRangeFeasible(feasibles):
    feasiblesN = dict()
    for key,feasible in feasibles.items():
        ranges = compressRange(feasible)
        feasiblesN[key] = ranges
    return feasiblesN

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