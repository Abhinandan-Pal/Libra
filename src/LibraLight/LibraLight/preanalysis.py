from colorama import Style, Fore
# from pip._vendor.colorama import Style

from abstract_domain import init, analyze
import config
from libra2others import chunk2json


def worker1(id, color, queue1, nnet, spec, shared):
    inputs, num_relus, layers, outputs = nnet
    sensitive, values, splittable = spec
    patterns, partitions, discarded, min_difference, difference, unstable, max_unstable, fair, biased, feasible, explored, json, lock = shared
    while True:
        percent, steps, size, disjuncts, ranges = queue1.get(block=True)
        if percent is None:  # no more chunks
            queue1.put((None, None, None, None, None))
            break
        #
        r_ranges = 'Ranges: {}'.format('; '.join('{} ∈ [{}, {}]'.format(i, l, u) for i, (l, u) in ranges if l != u))
        level, start = 0, percent
        while start < 100:
            level = level + 1
            start = percent * 2 ** level
        r_partition = '{} | Percent: {}, Level: {}'.format(r_ranges, percent, level)
        print(color + r_partition, Style.RESET_ALL)
        #
        ranges0 = dict(ranges)
        ranges0[sensitive] = (values[0][0], values[1][1])
        initial = init(ranges0)
        active, inactive, outcome, (polarity, symbols) = analyze(initial, inputs, layers, outputs)
        #
        if outcome:      # ✔︎ No Bias
            print(Fore.GREEN + '✔︎ No Bias ({}) in {}'.format(outcome, r_partition), Style.RESET_ALL)
            lock.acquire()
            partitions.value += 1
            feasible.value += percent
            explored.value += percent
            if explored.value >= 100:
                queue1.put((None, None, None, None, None))
            discarded.value += 1
            fair.value += percent
            curr = json.get('fair', list())
            chunk = '; '.join('{} ∈ [{}, {}]'.format(i, l, u) for i, (l, u) in ranges)
            jsoned = chunk2json(chunk)
            jsoned['outcome'] = outcome
            curr.append(jsoned)
            json['fair'] = curr
            lock.release()
            progress = 'Progress for #{}: {}% of {}% ({}% fair)'.format(id, feasible.value, explored.value, fair.value)
            print(Fore.YELLOW + progress, Style.RESET_ALL)
        else:
            disjunctions = num_relus - len(active) - len(inactive)
            if disjunctions <= disjuncts:     # ‼ Possible Bias
                key = (frozenset(active), frozenset(inactive))
                value = (percent, (frozenset(ranges)))
                lock.acquire()
                partitions.value += 1
                feasible.value += percent
                explored.value += percent
                if explored.value >= 100:
                    queue1.put((None, None, None, None, None))
                curr = patterns.get(key, set())
                curr.add(value)
                patterns[key] = curr
                lock.release()
                found = '‼ Possible Bias in {}'.format(r_partition)
                print(Fore.LIGHTYELLOW_EX + found, Style.RESET_ALL)
                progress = 'Progress for #{}: {}% of {}% ({}% fair)'.format(id, feasible.value, explored.value, fair.value)
                print(Fore.YELLOW + progress, Style.RESET_ALL)
            else:   # Too many disjunctions
                print('Too many disjunctions ({})!'.format(disjunctions))
                # ranges0 = dict(ranges)
                assert polarity
                if config.splitting == config.SplittingHeuristic.RELU_POLARITY:
                    combined = {key: abs(value) for key, value in symbols[0].items() if key in splittable}
                else:
                    assert config.splitting == config.SplittingHeuristic.OUTPUT_POLARITY
                    combined = {key: abs(value) for key, value in symbols.items() if key in splittable}
                splitters = sorted(combined.items(), key=lambda x: x[1], reverse=True)
                splitters = [splitter[0] for splitter in splitters]
                #
                splitter = splitters[0]
                (lower, upper) = ranges0[splitter]
                while splitters and upper - lower <= size:
                    print('Cannot range split for {} anymore!'.format(splitter))
                    splitters = splitters[1:]
                    if splitters:
                        splitter = splitters[0]
                        (lower, upper) = ranges0[splitter]
                if splitters and size < upper - lower:
                    middle = lower + (upper - lower) / 2
                    print('Range split for {} at: {}'.format(splitter, middle))
                    _percent = percent / 2
                    left = dict(ranges)
                    left[splitter] = (lower, middle)
                    _left = list(left.items())
                    right = dict(ranges)
                    right[splitter] = (middle, upper)
                    _right = list(right.items())
                    queue1.put((_percent, steps, size, disjuncts, _left))
                    queue1.put((_percent, steps, size, disjuncts, _right))
                elif 2 * min_difference <= size or disjuncts < max_unstable:   # autotuning
                    (stepsL, stepsU) = steps
                    if stepsU == 0 and disjuncts < max_unstable:
                        _stepsL, _stepsU = 0, 1
                        _size, _disjuncts = size, disjuncts + 1
                        print(Fore.BLUE + "Upper bound increase from: {} to: {}".format(disjuncts, _disjuncts), Style.RESET_ALL)
                    elif stepsU == 0 and 2 * min_difference <= size:
                        _stepsL, _stepsU = stepsL, stepsU
                        _size, _disjuncts = size / 2, disjuncts
                        print(Fore.BLUE + "Lower bound decrease from: {} to: {}".format(size, _size), Style.RESET_ALL)
                    elif stepsL == 0 and 2 * min_difference <= size:
                        _stepsL, _stepsU = 1, 0
                        _size, _disjuncts = size / 2, disjuncts
                        print(Fore.BLUE + "Lower bound decrease from: {} to: {}".format(size, _size), Style.RESET_ALL)
                    else:
                        assert stepsL == 0 and disjuncts < max_unstable
                        _stepsL, _stepsU = stepsL, stepsU
                        _size, _disjuncts = size, disjuncts + 1
                        print(Fore.BLUE + "Upper bound increase from: {} to: {}".format(disjuncts, _disjuncts), Style.RESET_ALL)
                    lock.acquire()
                    difference.value = min(difference.value, _size)
                    unstable.value = max(unstable.value, _disjuncts)
                    lock.release()
                    print(Fore.BLUE + "Autotuned to: L = {}, U = {}".format(_size, _disjuncts), Style.RESET_ALL)
                    queue1.put((percent, (_stepsL, _stepsU), _size, _disjuncts, ranges))
                else:
                    lock.acquire()
                    explored.value += percent
                    if explored.value >= 100:
                        queue1.put((None, None, None, None, None))
                    curr = json.get('unknown', list())
                    chunk = '; '.join('{} ∈ [{}, {}]'.format(i, l, u) for i, (l, u) in ranges)
                    curr.append(chunk2json(chunk))
                    json['unknown'] = curr
                    lock.release()
                    found = '‼ Unchecked Bias in {}'.format(r_partition)
                    print(Fore.RED + found, Style.RESET_ALL)
                    progress = 'Progress for #{}: {}% of {}% ({}% fair)'.format(id, feasible.value, explored.value, fair.value)
                    print(Fore.YELLOW + progress, Style.RESET_ALL)
