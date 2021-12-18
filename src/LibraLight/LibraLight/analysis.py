import ctypes
import itertools
from copy import deepcopy

import pulp
from apronpy.manager import FunId
from apronpy.var import PyVar
from colorama import Style, Fore

from libra2others import chunk2json
from polyhedra_domain import init_polyhedra, analyze_polyhedra, assume_ranges, to_apron, to_pulp, meet_apron_ranges, \
    from_apron, meet_polyhedra, assume_constants, length, simplify_polyhedra


def worker2(id, color, queue2, total, nnet, spec, shared):
    inputs, relus, layers, outputs = nnet
    sensitive, values, bounds = spec
    patterns, partitions, discarded, _, difference, unstable, _, fair, biased, feasible, explored, json, lock = shared
    while True:
        idx, (key, pack) = queue2.get(block=True)
        if idx is None:  # no more abstract activation patterns
            queue2.put((None, (None, None)))
            break
        print(color + 'Pattern #{} of {} [{}]'.format(idx, total, len(pack)), Style.RESET_ALL)
        active, inactive = key
        #
        pulpbounds = dict(bounds)
        constants = dict()
        for name, (lower, upper) in pulpbounds.items():
            if lower == upper:
                constants[name] = lower
        # print('constants', constants)
        check = dict()
        check[(outputs[0], values[0])] = list()
        check[(outputs[0], values[1])] = list()
        check[(outputs[1], values[0])] = list()
        check[(outputs[1], values[1])] = list()
        min_int = (-ctypes.c_uint(-1).value) // 2
        outcome0 = init_polyhedra(layers[-1], outputs[0], typ=0)
        flattened0 = itertools.chain.from_iterable(reversed([layer.items() for layer in layers[:-1]]))
        count0 = 0
        for state0 in analyze_polyhedra(outcome0, flattened0, active, inactive):
            _state0 = assume_constants(deepcopy(state0), constants)
            count0 += 1
            pulpbounds[sensitive] = values[0]
            _state00 = assume_ranges(deepcopy(_state0), {sensitive: values[0]})
            pulpstate00 = to_pulp(_state00, pulpbounds)
            if pulpstate00:
                # print(color + '0/{}/0 Simplifying...'.format(count0), length(_state00), Style.RESET_ALL)
                # __state00 = simplify_polyhedra(_state00, pulpbounds)
                print(color + '0/{}/0 Converting to APRON...'.format(count0), Style.RESET_ALL)
                apronstate00 = to_apron(_state00, inputs)
                print(color + '0/{}/0 Working in APRON...'.format(count0), Style.RESET_ALL)
                apronstate00.manager.manager.contents.option.funopt[FunId.AP_FUNID_FORGET_ARRAY].algorithm = min_int
                apronstate00 = apronstate00.forget([PyVar(sensitive)])
                print(color + '0/{}/0 Converting back from APRON...'.format(count0), Style.RESET_ALL)
                apronstate00.manager.manager.contents.option.funopt[FunId.AP_FUNID_TO_LINCONS_ARRAY].algorithm = min_int
                state00 = from_apron(apronstate00, [ipt for ipt in inputs if ipt != sensitive])
                check[(outputs[0], values[0])].append(state00)
            #
            pulpbounds[sensitive] = values[1]
            _state01 = assume_ranges(deepcopy(_state0), {sensitive: values[1]})
            pulpstate01 = to_pulp(_state01, pulpbounds)
            if pulpstate01:
                # print(color + '0/{}/1 Simplifying...'.format(count0), length(_state01), Style.RESET_ALL)
                # __state01 = simplify_polyhedra(_state01, pulpbounds)
                print(color + '0/{}/1 Converting to APRON...'.format(count0), Style.RESET_ALL)
                apronstate01 = to_apron(_state01, inputs)
                print(color + '0/{}/1 Working in APRON...'.format(count0), Style.RESET_ALL)
                apronstate01.manager.manager.contents.option.funopt[FunId.AP_FUNID_FORGET_ARRAY].algorithm = min_int
                apronstate01 = apronstate01.forget([PyVar(sensitive)])
                print(color + '0/{}/1 Converting back from APRON...'.format(count0), Style.RESET_ALL)
                apronstate01.manager.manager.contents.option.funopt[FunId.AP_FUNID_TO_LINCONS_ARRAY].algorithm = min_int
                state01 = from_apron(apronstate01, [ipt for ipt in inputs if ipt != sensitive])
                check[(outputs[0], values[1])].append(state01)
        outcome1 = init_polyhedra(layers[-1], outputs[1], typ=1)
        flattened1 = itertools.chain.from_iterable(reversed([layer.items() for layer in layers[:-1]]))
        count1 = 0
        for state1 in analyze_polyhedra(outcome1, flattened1, active, inactive):
            _state1 = assume_constants(deepcopy(state1), constants)
            count1 += 1
            pulpbounds[sensitive] = values[0]
            _state10 = assume_ranges(deepcopy(_state1), {sensitive: values[0]})
            pulpstate10 = to_pulp(_state10, pulpbounds)
            if pulpstate10:
                # print(color + '1/{}/0 Simplifying...'.format(count1), length(_state10). Style.RESET_ALL)
                # __state10 = simplify_polyhedra(_state10, pulpbounds)
                print(color + '1/{}/0 Converting to APRON...'.format(count1), Style.RESET_ALL)
                apronstate10 = to_apron(_state10, inputs)
                print(color + '1/{}/0 Working in APRON...'.format(count1), Style.RESET_ALL)
                apronstate10.manager.manager.contents.option.funopt[FunId.AP_FUNID_FORGET_ARRAY].algorithm = min_int
                apronstate10 = apronstate10.forget([PyVar(sensitive)])
                print(color + '1/{}/0 Converting back from APRON...'.format(count1), Style.RESET_ALL)
                apronstate10.manager.manager.contents.option.funopt[FunId.AP_FUNID_TO_LINCONS_ARRAY].algorithm = min_int
                state10 = from_apron(apronstate10, [ipt for ipt in inputs if ipt != sensitive])
                check[(outputs[1], values[0])].append(state10)
            #
            pulpbounds[sensitive] = values[1]
            _state11 = assume_ranges(deepcopy(_state1), {sensitive: values[1]})
            pulpstate11 = to_pulp(_state11, pulpbounds)
            if pulpstate11:
                # print(color + '1/{}/1 Simplifying...'.format(count1), length(_state11), Style.RESET_ALL)
                # __state11 = simplify_polyhedra(_state11, pulpbounds)
                print(color + '1/{}/1 Converting to APRON...'.format(count1), Style.RESET_ALL)
                apronstate11 = to_apron(_state11, inputs)
                print(color + '1/{}/1 Working in APRON...'.format(count1), Style.RESET_ALL)
                apronstate11.manager.manager.contents.option.funopt[FunId.AP_FUNID_FORGET_ARRAY].algorithm = min_int
                apronstate11 = apronstate11.forget([PyVar(sensitive)])
                print(color + '1/{}/1 Converting back from APRON...'.format(count1), Style.RESET_ALL)
                apronstate11.manager.manager.contents.option.funopt[FunId.AP_FUNID_TO_LINCONS_ARRAY].algorithm = min_int
                state11 = from_apron(apronstate11, [ipt for ipt in inputs if ipt != sensitive])
                check[(outputs[1], values[1])].append(state11)
        #
        print(color + 'Checking for bias...', Style.RESET_ALL)
        for (percent, ranges) in pack:
            r_ranges = 'Ranges: {}'.format('; '.join('{} ∈ [{}, {}]'.format(i, l, u) for i, (l, u) in ranges if l != u))
            #
            nobias = True
            b_ranges = dict()
            items = list(check.items())
            for i in range(len(items)):
                (outi, vali), statesi = items[i]
                for j in range(i+1, len(items)):
                    (outj, valj), statesj = items[j]
                    if outi != outj and vali != valj:
                        for statei in statesi:
                            for statej in statesj:
                                meet = meet_polyhedra(statei, statej)
                                pulpstate = to_pulp(meet, dict(ranges), log=True)
                                if pulpstate:
                                    _b_ranges = pulpstate[3]
                                    for v, (l, u) in _b_ranges.items():
                                        if v in b_ranges:
                                            inf, sup = b_ranges[v]
                                            b_ranges[v] = (min(l, inf), max(u, sup))
                                        else:
                                            b_ranges[v] = (l, u)
                                    nobias = False
                                    sensitivei = '{} ∈ [{}, {}]'.format(sensitive, vali[0], vali[1])
                                    sensitivej = '{} ∈ [{}, {}]'.format(sensitive, valj[0], valj[1])
                                    pair = '{} -> {} vs {} -> {}'.format(sensitivei, outi, sensitivej, outj)
                                    found = '✘ Bias Found ({})! in {}'.format(pair, r_ranges)
                                    cex = pulpstate[2]
                                    cex['sensitive1'] = sensitivei
                                    cex['outcome1'] = outi
                                    cex['sensitive2'] = sensitivej
                                    cex['outcome2'] = outj
                                    counterexample = {
                                        'sensitive1': sensitivei,
                                        'outcome1': outi,
                                        'sensitive2': sensitivej,
                                        'outcome2': outj
                                    }
                                    lock.acquire()
                                    curr = json.get('biased', list())
                                    chunk = '; '.join('{} ∈ [{}, {}]'.format(i, l, u) for i, (l, u) in ranges)
                                    jsoned = chunk2json(chunk)
                                    jsoned['counterexample'] = cex
                                    jsoned['constraints'] = str(pulpstate[1].constraints)
                                    curr.append(jsoned)
                                    json['biased'] = curr
                                    lock.release()
                                    print(Fore.RED + found, Style.RESET_ALL)
            if nobias:
                outcomes = set()
                for i in range(len(items)):
                    (outi, vali), statesi = items[i]
                    for statei in statesi:
                        conj = dict(ranges)
                        pulpstate = to_pulp(statei, conj)
                        if pulpstate:
                            outcomes.add(outi)
                classes = ', '.join(str(outcome) for outcome in outcomes)
                print(Fore.GREEN + '✔︎ No Bias ({}) in {}'.format(classes, r_ranges), Style.RESET_ALL)
                lock.acquire()
                curr = json.get('fair', list())
                chunk = '; '.join('{} ∈ [{}, {}]'.format(i, l, u) for i, (l, u) in ranges)
                curr.append(chunk2json(chunk))
                json['fair'] = curr
                lock.release()
            else:
                total_size = 1
                for _, (lower, upper) in ranges:
                    if upper - lower > 0:
                        total_size *= upper - lower
                biased_size = 1
                for (lower, upper) in b_ranges.values():
                    if upper - lower > 0:
                        biased_size *= upper - lower
                _percent = percent * biased_size / total_size
                lock.acquire()
                biased.value += _percent
                lock.release()


if __name__ == '__main__':
    solver_list = pulp.listSolvers(onlyAvailable=True)
    print(solver_list)
    x = pulp.LpVariable("x", 0.0, 1, pulp.LpContinuous)
    y = pulp.LpVariable("y", 0.0, 1, pulp.LpContinuous)
    problem = pulp.LpProblem("", pulp.const.LpMaximize)
    problem += x + 2 * y, ""
    problem += -4 * x + 2 * y >= 1, "a"
    problem += -4 * x + y >= 0, "b"
    problem += y >= - x, "c"
    problem.solve()
    print(problem.unusedConstraintName())
    print(pulp.value(problem.objective))
    print(problem.toDict())
    # #
    # obj = pulp.LpConstraintVar("x")
    # problem.setObjective(obj)
    # problem.sense = pulp.const.LpMinimize
    # problem.solve()
    # for variable in problem.variables():
    #     if variable.name == "x":
    #         lower = variable.varValue
    #         print(lower)
    # # lower = pulp.value(problem.objective)
    # problem.sense = pulp.const.LpMaximize
    # problem.solve()
    # for variable in problem.variables():
    #     if variable.name == "x":
    #         upper = variable.varValue
    #         print(upper)
    # print('x', lower, upper)
    #
    # obj = pulp.LpConstraintVar("y")
    # problem.setObjective(obj)
    # problem.sense = pulp.const.LpMinimize
    # problem.solve()
    # lower = pulp.value(problem.objective)
    # problem.sense = pulp.const.LpMaximize
    # problem.solve()
    # upper = pulp.value(problem.objective)
    # print('y', lower, upper)
