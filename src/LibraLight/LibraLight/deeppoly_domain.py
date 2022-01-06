from copy import deepcopy
from typing import Dict


def init_deeppoly(ranges):
    bounds = dict()
    for ipt, val in ranges.items():
        bounds[ipt] = val
    poly = dict()
    for ipt in bounds.keys():
        lower = dict()
        lower['_'] = bounds[ipt][0]
        upper = dict()
        upper['_'] = bounds[ipt][1]
        poly[ipt] = (lower, upper)
    expressions = dict()
    polarities = dict()
    flags = dict()
    return bounds, poly, expressions, polarities, flags


def is_bottom(deeppoly):
    bounds, _, _, _, _ = deeppoly
    return any(val[0] > val[1] for val in bounds.values())


def evaluate(dictionary, bounds):
    result = (0, 0)
    for var, val in dictionary.items():
        coeff = (val, val)
        if var != '_':
            ac = 0 if coeff[0] == 0 or bounds[var][0] == 0 else coeff[0] * bounds[var][0]
            ad = 0 if coeff[0] == 0 or bounds[var][1] == 0 else coeff[0] * bounds[var][1]
            bc = 0 if coeff[1] == 0 or bounds[var][0] == 0 else coeff[1] * bounds[var][0]
            bd = 0 if coeff[1] == 0 or bounds[var][1] == 0 else coeff[1] * bounds[var][1]
            result = (result[0] + min(ac, ad, bc, bd), result[1] + max(ac, ad, bc, bd))
        else:
            result = (result[0] + coeff[0], result[1] + coeff[1])
    return result


def affine(deeppoly, inputs, layer):
    if is_bottom(deeppoly):
        return deeppoly
    bounds, poly, expressions, polarities, flags = deepcopy(deeppoly)
    for lhs, rhs in layer.items():
        _inf, inf = deepcopy(rhs), deepcopy(rhs)
        _sup, sup = deepcopy(rhs), deepcopy(rhs)
        poly[lhs] = (_inf, _sup)
        while any(variable in inf and variable not in inputs for variable in poly):
            for variable in poly:
                if variable in inf and variable not in inputs:  # should be replaced
                    coeff = inf[variable]
                    if coeff > 0:
                        replacement = poly[variable][0]
                    elif coeff < 0:
                        replacement = poly[variable][1]
                    else:  # coeff == 0
                        replacement = dict()
                        replacement['_'] = 0.0
                    del inf[variable]
                    for var, val in replacement.items():
                        if var in inf:
                            inf[var] += coeff * val
                        else:
                            inf[var] = coeff * val
        while any(variable in sup and variable not in inputs for variable in poly):
            for variable in poly:
                if variable in sup and variable not in inputs:  # should be replaced
                    coeff = sup[variable]
                    if coeff > 0:
                        replacement = poly[variable][1]
                    elif coeff < 0:
                        replacement = poly[variable][0]
                    else:  # coeff == 0
                        replacement = dict()
                        replacement['_'] = 0.0
                    del sup[variable]
                    for var, val in replacement.items():
                        if var in sup:
                            sup[var] += coeff * val
                        else:
                            sup[var] = coeff * val
        lower = evaluate(inf, bounds)[0]
        upper = evaluate(sup, bounds)[1]
        bounds[lhs] = (lower, upper)
        if lower < 0 and 0 < upper:
            expressions[lhs] = (inf, sup)
            polarities[lhs] = abs((lower + upper) / (upper - lower))
    return bounds, poly, expressions, polarities, flags


def relu(deeppoly, layer):
    if is_bottom(deeppoly):
        return deeppoly
    bounds, poly, expressions, polarities, flags = deepcopy(deeppoly)
    for lhs in layer:
        lower, upper = bounds[lhs]
        if upper <= 0:
            bounds[lhs] = (0, 0)
            zero: Dict[str, float] = dict()
            zero['_'] = 0.0
            poly[lhs] = (zero, zero)
            flags[lhs] = -1
        elif 0 <= lower:
            flags[lhs] = 1
        else:
            if upper <= -lower:  # case (b) in Fig. 4, equation (3)
                bounds[lhs] = (0, upper)
                inf: Dict[str, float] = dict()
                inf['_'] = 0.0
            else:  # case (c) in Fig. 4, equation (3)
                bounds[lhs] = (lower, upper)
                inf = deepcopy(poly[lhs][0])
            m = upper / (upper - lower)
            if m > 0:
                sup = poly[lhs][1]
            elif m < 0:
                sup = poly[lhs][0]
            else:  # m == 0
                sup = dict()
                sup['_'] = 0.0
            for var, val in sup.items():
                sup[var] = m * val
            q = - upper * lower / (upper - lower)
            sup['_'] = sup['_'] + q
            poly[lhs] = (inf, sup)
            flags[lhs] = 0
        if is_bottom((bounds, poly, expressions, polarities, flags)):
            flags[lhs] = -1
    return bounds, poly, expressions, polarities, flags


def outcome(deeppoly, inputs, outputs):
    if is_bottom(deeppoly):
        return '⊥'
    else:
        bounds, poly, expressions, polarities, flags = deepcopy(deeppoly)
        (lower1, upper1) = bounds[outputs[0]]
        (lower2, upper2) = bounds[outputs[1]]
        # z = output0 - output1
        out0 = dict()
        out0[outputs[0]] = 1
        out0[outputs[1]] = -1
        while any(variable in out0 and variable not in inputs for variable in poly):
            # print(exp)
            for variable in poly:
                if variable in out0 and variable not in inputs:  # should be replaced
                    coeff = out0[variable]
                    if coeff > 0:
                        replacement = poly[variable][0]
                    elif coeff < 0:
                        replacement = poly[variable][1]
                    else:  # coeff == 0
                        replacement = dict()
                        replacement['_'] = 0.0
                    del out0[variable]
                    for var, val in replacement.items():
                        if var in out0:
                            out0[var] += coeff * val
                        else:
                            out0[var] = coeff * val
        lower1, upper1 = evaluate(out0, bounds)
        if 0 <= lower1:   # upper2 < lower1:
            return "<=", (None, None)
        else:
            # z = output1 - output0
            out1 = dict()
            out1[outputs[0]] = -1
            out1[outputs[1]] = 1
            while any(variable in out1 and variable not in inputs for variable in poly):
                # print(exp)
                for variable in poly:
                    if variable in out1 and variable not in inputs:  # should be replaced
                        coeff = out1[variable]
                        if coeff > 0:
                            replacement = poly[variable][0]
                        elif coeff < 0:
                            replacement = poly[variable][1]
                        else:  # coeff == 0
                            replacement = dict()
                            replacement['_'] = 0.0
                        del out1[variable]
                        for var, val in replacement.items():
                            if var in out1:
                                out1[var] += coeff * val
                            else:
                                out1[var] = coeff * val
            lower2, upper2 = evaluate(out1, bounds)
            if 0 < lower2:
                return ">", (None, None)
            else:
                polarity1 = abs((lower1 + upper1) / (upper1 - lower1))
                polarity2 = abs((lower2 + upper2) / (upper2 - lower2))
                if polarity1 <= polarity2:
                    return None, (polarity1, out0)
                else:
                    return None, (polarity2, out1)


def analyze_deeppoly(deeppoly, inputs, layers, outputs):
    current = deeppoly
    for layer in layers[:-1]:
        current = relu(affine(current, inputs, layer), layer)
    final = affine(current, inputs, layers[-1])
    bounds, poly, expressions, polarities, flags = final
    activated = {lhs for lhs, flag in flags.items() if flag == 1}
    deactivated = {lhs for lhs, flag in flags.items() if flag == -1}
    found, (polarity, expression) = outcome(final, inputs, outputs)
    #print(f"\tBounds-> x50: {bounds['x50']} x51: {bounds['x51']}")
    #print(f"\tBounds-> {bounds}")
    #print(f"\tOutcome-> {found}")
    import config
    if config.splitting == config.SplittingHeuristic.RELU_POLARITY:
        polarity = min(polarities) if polarities else None
        symbols = expressions[min(polarities, key=polarities.get)] if polarities else None
        return activated, deactivated, found, (polarity, symbols)
    else:
        assert config.splitting == config.SplittingHeuristic.OUTPUT_POLARITY
        return activated, deactivated, found, (polarity, expression)
