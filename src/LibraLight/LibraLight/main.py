import json
import random
import time
from multiprocessing import Manager, Value, set_start_method, Lock
from multiprocessing.context import Process
import numpy as np
from colorama import Fore, Style
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from analysis import worker2
from keras2libra import parse_keras
from preanalysis import worker1
import config
from optimized import deeppoly_gpu as dpG
from optimized import symbolic_gpu as symG
from optimized import neurify_gpu as neuG
from optimized import product_gpu as prodG
from preanalysis_gpu import preanalysis as preAG
import test

preanalysis_time = None
analysis_time = None


def set_domain(mins, maxs):
    config.bounds = dict()
    config.continuous = list()
    for i, val in enumerate(zip(mins, maxs)):
        name = 'x0{}'.format(i)
        config.bounds[name] = val
        if val[1] - val[0] > 0:
            config.continuous.append(name)


def set_sensitive(sensitive):
    config.sensitive = 'x0{}'.format(sensitive)
    lower, upper = config.bounds[config.sensitive][0], config.bounds[config.sensitive][1]
    middle = lower + (upper - lower) / 2
    config.values = ((lower, middle), (middle, upper))
    del config.bounds[config.sensitive]
    config.continuous.remove(config.sensitive)

def set_sensitive_GPU(sensitive):
    config.sensitive = 'x0{}'.format(sensitive)
    lower, upper = config.bounds[config.sensitive][0], config.bounds[config.sensitive][1]
    middle = lower + (upper - lower) / 2
    config.values = ((lower, middle), (middle, upper))
    config.continuous.remove(config.sensitive)

def print_result(result, time1, time2,ifGPU,config,XMAX):
    _, _, _, _, _, _, _, fair, biased, feasible, _, _, _ = result
    print('Analyzed Input Domain Percentage: {}%'.format(feasible.value))
    print('Certified Fair Percentage: {}%'.format(fair.value))
    print('Potentially Biased Percentage: {}%'.format(biased.value))
    print('Uncertified Input Domain Percentage: {}%'.format(100 - feasible.value))
    print('Pre-Analysis Time: {}s'.format(time1))
    print('Analysis Time: {}s'.format(time2))
    file1 = open(f'jsonFiles/{ifGPU}.txt', 'a')

    file1.write(f"\n\nTHRESHOLD: {config.threshold} XMAX: {XMAX[7:]}\n")
    file1.write('Analyzed Input Domain Percentage: {}% \n'.format(feasible.value))
    file1.write('Certified Fair Percentage: {}% \n'.format(fair.value))
    file1.write('Potentially Biased Percentage: {}% \n'.format(biased.value))
    file1.write('Uncertified Input Domain Percentage: {}% \n'.format(100 - feasible.value))
    file1.write('Pre-Analysis Time: {}s \n'.format(time1))
    file1.write('Analysis Time: {}s \n'.format(time2))

    file1.close()

def preanalysis(shared):
    # prepare the queue
    queue1 = Manager().Queue()
    _percent = 100
    _steps = (0, 0)
    _size = config.start_difference
    _disjuncts = config.start_unstable
    _ranges = list(config.bounds.items())
    queue1.put((_percent, _steps, _size, _disjuncts, _ranges))
    nnet = (config.inputs, len(config.activations), config.layers, config.outputs)
    spec = (config.sensitive, config.values, config.continuous)
    patterns, partitions, discarded, min_difference, difference, unstable, max_unstable, fair, _, feasible, _, json_out, _ = shared
    # run the pre-analysis
    start1 = time.time()
    processes = list()
    for i in range(config.cpu):
        color = config.colors[i % len(config.colors)]
        args = (i, color, queue1, nnet, spec, shared)
        process = Process(target=worker1, args=args)
        processes.append(process)
        process.start()
    for process in processes:
        process.join()
    end1 = time.time()
    if min_difference < config.start_difference or config.start_unstable < max_unstable:
        print(Fore.BLUE + "\nAutotuned to: L = {}, U = {}".format(difference.value, unstable.value), Style.RESET_ALL)
    partitions = partitions.value
    discarded = discarded.value
    considered = partitions - discarded
    print(Fore.BLUE + '\nFound: {} patterns for {}[{}] partitions'.format(len(patterns), considered, partitions))
    #
    prioritized = sorted(patterns.items(), key=lambda v: len(v[1]), reverse=True)
    for key, pack in prioritized:
        sset = lambda s: '{{{}}}'.format(', '.join('{}'.format(e) for e in s))
        print(sset(key[0]), '|', sset(key[1]), '->', len(pack))
    #
    print(f"{patterns}")
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

    def max_disj(key):
        return len(config.activations) - len(key[0]) - len(key[1])

    prioritized = sorted(compressed.items(), key=lambda v: max_disj(v[0]) + len(v[1]), reverse=True)
    if len(compressed) < len(patterns):
        print('Compressed to: {} patterns'.format(len(compressed)))
        for key, pack in prioritized:
            sset = lambda s: '{{{}}}'.format(', '.join('{}'.format(e) for e in s))
            print(sset(key[0]), '|', sset(key[1]), '->', len(pack))
    #
    result = '\nPre-Analysis Result: {}% fair ({}% feasible)'.format(fair.value, feasible.value)
    print(Fore.BLUE + result)
    global preanalysis_time
    preanalysis_time = end1 - start1
    print('Pre-Analysis Time: {}s'.format(preanalysis_time), Style.RESET_ALL)
    return prioritized, preanalysis_time


def analysis(prioritized, shared):
    # prepare the queue
    queue2 = Manager().Queue()
    for idx, (key, pack) in enumerate(prioritized):
        queue2.put((idx + 1, (key, pack)))
    queue2.put((None, (None, None)))
    nnet = (config.inputs, config.activations, config.layers, config.outputs)
    spec = (config.sensitive, config.values, config.bounds)
    patterns, partitions, discarded, min_difference, difference, unstable, max_unstable, fair, biased, feasible, explored, json_out, lock = shared
    # run the analysis
    start2 = time.time()
    processes = list()
    for i in range(config.cpu):
        color = config.colors[i % len(config.colors)]
        args = (i, color, queue2, len(prioritized), nnet, spec, shared)
        process = Process(target=worker2, args=args)
        processes.append(process)
        process.start()
    for process in processes:
        process.join()
    end2 = time.time()
    #
    result = '\nResult: {}% of {}% ({}% biased)'.format(feasible.value, explored.value, biased.value)
    print(Fore.BLUE + result)
    print('Pre-Analysis Time: {}s'.format(preanalysis_time))
    global analysis_time
    analysis_time = end2 - start2
    print('Analysis Time: {}s'.format(analysis_time), Style.RESET_ALL)
    print('\nDone!')
    return json_out.copy(), analysis_time

def combineJSON(json1,json2):
    from collections import defaultdict
    jsonD = defaultdict(list)

    for js in (json1,json2):  # you can list as many input dicts as you want here
        for key, value in js.items():
            jsonD[key]+=(value)
    return jsonD

def do(out_name,ifGPU,domains):
    print(Fore.BLUE + '\n||==================================||')
    print('|| domain: {}'.format(config.domain))
    print('|| min_difference: {}'.format(config.min_difference))
    print('|| start_difference: {}'.format(config.start_difference))
    print('|| start_unstable: {}'.format(config.start_unstable))
    print('|| max_unstable: {}'.format(config.max_unstable))
    print('|| cpus: {}'.format(config.cpu))
    print('||==================================||', Style.RESET_ALL)

    patterns = Manager().dict()  # packing of abstract activation patterns
    partitions = Value('i', 0)  # number of input partitions
    discarded = Value('i', 0)  # number of discarded input partitions
    difference = Value('d', 2)  # current L
    unstable = Value('i', 0)  # current U
    fair = Value('d', 0.0)  # percentage that is fair
    biased = Value('d', 0.0)  # percentage that is biased
    feasible = Value('d', 0.0)  # percentage that could be analyzed
    explored = Value('d', 0.0)  # percentage that was explored
    json_out1 = dict()
    json_out = Manager().dict()
    shared = (
    patterns, partitions, discarded, config.min_difference, difference, unstable, config.max_unstable, fair, biased,
    feasible, explored, json_out, Lock())

    print(Fore.BLUE + '\n||==============||')
    print('|| Pre-Analysis ||')
    print('||==============||\n', Style.RESET_ALL)

    if(ifGPU):
        json_out1,prioritized, time1,feasible,fair = preAG(json_out1,config, config.start_difference, config.min_difference, config.max_unstable,domains)
        fair = Value('d', fair)  # percentage that is fair
        feasible = Value('d', feasible)  # percentage that could be analyzed
        shared = (patterns, partitions, discarded, config.min_difference, difference, unstable, config.max_unstable, fair,biased,feasible, explored, json_out, Lock())
    else:
        prioritized, time1 = preanalysis(shared)
        patterns, partitions, discarded, min_difference, difference, unstable, max_unstable, fair, biased, feasible, explored, json_out, lock = shared



    print(f"Prioritized : {prioritized}\n Time: {time1}")

    print('||==========||\n', Style.RESET_ALL)

    if(ifGPU):
        result,time2 = json_out1.copy(),0.0
    else:
        result, time2 = json_out.copy(), 0.0
    #result, time2 = analysis(prioritized, shared)
    #result = combineJSON(json_out1, result)

    minL, startL = config.min_difference, config.start_difference
    startU, maxU = config.start_unstable, config.max_unstable
    out_file = 'jsonFiles/result-{}_{}_{}-{}_{}-{}-{}.json'.format(out_name, config.threshold, minL, startL, startU, maxU,ifGPU)
    with open(out_file, 'w', encoding='utf8') as f:
        json.dump(result, f, indent=4, separators=(',', ':'), ensure_ascii=False)
    return shared, time1, time2


def test1(ifGPU,domains):
    def perfom(t,bnds):
        bnd = []
        for i in range(0, len(bnds)):
            bnd.append(int(bnds[i]))
        config.threshold = t
        toy_model = Sequential()
        toy_model.add(Dense(12, activation='relu'))
        toy_model.add(Dense(12, activation='relu'))
        toy_model.add(Dense(12, activation='relu'))
        toy_model.add(Dense(12, activation='relu'))
        toy_model.add(Dense(1))
        toy_model.compile('adam', 'mse')
        toy_model.predict(np.ones((1, 12)))
        toy_model.load_weights('toy.hd5')
        inputs, activations, layers, outputs = parse_keras(toy_model, config.threshold)
        config.inputs = inputs
        config.activations = activations
        config.layers = layers
        config.outputs = outputs

        MIN_ = np.array([0, 40.000, -73.000, -2000.000, 20.018, -45, -2, 0, 0, 0, 0, 0])
        MAX_ = np.array([1, 105.000, 40.000, 15100.000, 219.985, 15.000, 2.000, 1, 1, 1, 1, 1])
        X_min = [0, 40, -73, -2000, 20.018, -45, 2, bnd[0], bnd[1], bnd[2], bnd[3],
                 bnd[4]]  # last 5 here 11011/10011/00111/01011/[][][]00    -2
        X_max = [1, 105, 40, 15100, 219.985, 15, 2, bnd[0], bnd[1], bnd[2], bnd[3], bnd[4]]
        c_min = (MAX_ - X_min) / (MAX_ - MIN_)
        c_max = (MAX_ - X_max) / (MAX_ - MIN_)
        x_min = -c_min + (1 - c_min)
        x_max = -c_max + (1 - c_max)
        set_domain(x_min, x_max)

        if (ifGPU):
            set_sensitive_GPU(0)
        else:
            set_sensitive(0)

        config.min_difference = 0.25
        config.start_difference = 2
        config.start_unstable = 2
        config.max_unstable = 2

        result, time1, time2 = do('test1', ifGPU,domains)
        print_result(result, time1, time2, ifGPU, config, X_max)

    '''for t in range(2540,2560):
        for bnds in ("11011","10011","00111","01011","11000","10000","00100","01000"):
                perfom(t,bnds)'''
    perfom(2550, "00011")



def toy(ifGPU,domains):
    config.inputs = ['x01', 'x02']
    config.activations = ['x11', 'x12', 'x21', 'x22']
    config.layers = [
        {
            'x11': {'x01': -0.31, 'x02': 0.99, '_': -0.63},
            'x12': {'x01': -1.25, 'x02': -0.64, '_': 1.88}
        },
        {
            'x21': {'x11': 0.4, 'x12': 1.21, '_': 0.00},
            'x22': {'x11': 0.64, 'x12': 0.69, '_': -0.39}
        },
        {
            'x31': {'x21': 0.26, 'x22': 0.33, '_': 0.45},
            'x32': {'x21': 1.42, 'x22': 0.40, '_': -0.45}
        }
    ]
    config.outputs = ('x31', 'x32')

    config.bounds = {'x01': (0, 1), 'x02': (0, 1)}
    config.continuous = ['x01', 'x02']
    if(ifGPU):
        set_sensitive_GPU(2)
    else:
        set_sensitive(2)

    config.min_difference = 0.25
    config.start_difference = 1
    config.start_unstable = 2
    config.max_unstable = 2

    result, time1, time2 = do('toy',ifGPU,domains)
    #print_result(result, time1, time2)

    # how to create initial states
    # 1. create one or more ranges dictionary:
    # the variable whose name is equal to config.sensitive must always range (config.values[0][0], config.values[1][1])
    # all other variables can be split in ranges of size L
    # e.g., L = 1
    '''ranges1 = dict()
    ranges1[config.sensitive] = (config.values[0][0], config.values[1][1])
    ranges1['x01'] = (-1, 0)
    ranges2 = dict()
    ranges2['x01'] = (0, 1)
    ranges2[config.sensitive] = (config.values[0][0], config.values[1][1])
    # 2. call init with the ranges dictionary
    from abstract_domain import init
    initial1 = init(ranges1)
    initial2 = init(ranges2)
    print(f"{ranges2}")
    #symG.analyze(initial1, config.inputs, config.layers, config.outputs)
    #prodG.analyze(initial1, config.inputs, config.layers, config.outputs, domains={"Symbolic"})
    #time_sec = time.time()
    #prodG.analyze(initial2,config.sensitive, config.inputs, config.layers, config.outputs, domains={"DeepPoly","Symbolic","Neurify"})
    #time_sec = time.time() - time_sec
    #print(f"GPU time: {time_sec}\n\n")
    # we can now run the forward analysis for each initial
    from abstract_domain import analyze
    active1, inactive1, outcome1, (polarity1, symbols1) = analyze(initial1, config.inputs, config.layers, config.outputs)
    print('active1:', active1)
    print('inactive1:', inactive1)
    print('outcome1:', outcome1)
    active2, inactive2, outcome2, (polarity2, symbols2) = analyze(initial2, config.inputs, config.layers, config.outputs)
    print('active2:', active2)
    print('inactive2:', inactive2)
    print('outcome2:', outcome2)
    # (just printing the results for now, we should later do something with these results, I will tell you)

    # the idea is to be able to run the above analysis in parallel for MANY initials as discussed'''


if __name__ == '__main__':
    set_start_method("fork", force=True)
    ifGPU = True
    domains = ["DeepPoly","Symbolic","Neurify"]
    toy(ifGPU,domains)
    #test.toy()
    #test1(False,domains)
    test1(True,domains)
