import ctypes
import itertools
from copy import deepcopy

from apronpy.coeff import PyMPQScalarCoeff
from apronpy.environment import PyEnvironment
from apronpy.lincons0 import ConsTyp
from apronpy.lincons1 import PyLincons1, PyLincons1Array
from apronpy.linexpr1 import PyLinexpr1
from apronpy.manager import FunId
from apronpy.polka import PyPolka, PyPolkaMPQstrictManager
from apronpy.tcons1 import PyTcons1Array, PyTcons1
from apronpy.texpr0 import TexprOp, TexprRtype, TexprRdir
from apronpy.texpr1 import PyTexpr1
from apronpy.var import PyVar
from pulp import pulp, PULP_CBC_CMD, LpConstraintVar


def init_polyhedra(out_layer, chosen, typ=0):
    polyhedra = {ConsTyp.AP_CONS_SUPEQ: list(), ConsTyp.AP_CONS_SUP: list()}
    constraint, other = None, None
    for out, expr in out_layer.items():
        if out == chosen:
            constraint = deepcopy(expr)
        else:
            other = deepcopy(expr)
    # chosen > other
    for var, val in other.items():
        constraint[var] = constraint[var] - val
    if typ:
        polyhedra[ConsTyp.AP_CONS_SUP].append(constraint)
    else:
        polyhedra[ConsTyp.AP_CONS_SUPEQ].append(constraint)
    return polyhedra


def length(polyhedra):
    return len(polyhedra[ConsTyp.AP_CONS_SUPEQ]) + len(polyhedra[ConsTyp.AP_CONS_SUP])


def meet_polyhedra(polyhedra1, polyhedra2):
    polyhedra = deepcopy(polyhedra1)
    for typ, constraints in deepcopy(polyhedra2).items():
        polyhedra[typ].extend(constraints)
    return polyhedra


def assume_ranges(polyhedra, ranges):
    for variable, bounds in ranges.items():
        lower = {variable: 1, '_': - bounds[0]} if bounds[0] != 0.0 else {variable: 1, '_': 0.0}
        polyhedra[ConsTyp.AP_CONS_SUPEQ].append(lower)
        upper = {variable: -1, '_': bounds[1]}
        polyhedra[ConsTyp.AP_CONS_SUPEQ].append(upper)
    return polyhedra


def assume_constants(polyhedra, values):
    for constraint in itertools.chain(polyhedra[ConsTyp.AP_CONS_SUPEQ], polyhedra[ConsTyp.AP_CONS_SUP]):
        # print(constraint)
        for variable, value in values.items():
            coeff = constraint[variable]
            del constraint[variable]
            constraint['_'] += coeff * value
    return polyhedra


def meet_apron_ranges(apron, ranges, inputs):
    environment = PyEnvironment([], [PyVar(ipt) for ipt in inputs])
    array = list()
    for name, bounds in ranges.items():
        lower = PyLinexpr1(environment)
        lower.set_coeff(PyVar(name), PyMPQScalarCoeff(float(1)))
        lower.set_cst(PyMPQScalarCoeff(float(- bounds[0])))
        array.append(PyLincons1(ConsTyp.AP_CONS_SUPEQ, lower))
        upper = PyLinexpr1(environment)
        upper.set_coeff(PyVar(name), PyMPQScalarCoeff(float(-1)))
        upper.set_cst(PyMPQScalarCoeff(float(bounds[1])))
        array.append(PyLincons1(ConsTyp.AP_CONS_SUPEQ, upper))
    a = PyLincons1Array(array)
    print('a', len(a))
    manager = PyPolkaMPQstrictManager()
    min_int = (-ctypes.c_uint(-1).value) // 2
    manager.manager.contents.option.funopt[FunId.AP_FUNID_MEET].algorithm = min_int
    manager.manager.contents.option.funopt[FunId.AP_FUNID_MEET_ARRAY].algorithm = min_int
    manager.manager.contents.option.funopt[FunId.AP_FUNID_MEET_LINCONS_ARRAY].algorithm = min_int
    abstract1 = PyPolka(manager, environment, array=a)
    print('abstract1', abstract1)
    meet = apron.meet(abstract1)
    print('meet', meet)
    representation = repr(meet)
    if not representation.startswith('-1.0 >= 0') and not representation == '⊥':
        return meet
    else:
        return None


def to_apron(polyhedra, inputs):
    environment = PyEnvironment([], [PyVar(ipt) for ipt in inputs])
    array = list()
    for typ, constraints in polyhedra.items():
        for constraint in constraints:
            expr = PyLinexpr1(environment)
            for name, coeff in constraint.items():
                if name != '_':
                    expr.set_coeff(PyVar(name), PyMPQScalarCoeff(float(coeff)))
                else:
                    expr.set_cst(PyMPQScalarCoeff(float(coeff)))
            array.append(PyLincons1(typ, expr))
    a = PyLincons1Array(array)
    # print('a', len(a))
    manager = PyPolkaMPQstrictManager()
    min_int = (-ctypes.c_uint(-1).value) // 2
    manager.manager.contents.option.funopt[FunId.AP_FUNID_MEET].algorithm = min_int
    manager.manager.contents.option.funopt[FunId.AP_FUNID_MEET_ARRAY].algorithm = min_int
    manager.manager.contents.option.funopt[FunId.AP_FUNID_MEET_LINCONS_ARRAY].algorithm = min_int
    abstract1 = PyPolka(manager, environment, array=a)
    # print('abstract1', abstract1)
    representation = repr(abstract1)
    if not representation.startswith('-1.0 >= 0') and not representation == '⊥':
        return abstract1
    else:
        return None


def from_apron(abstract1, inputs):
    polyhedra = {ConsTyp.AP_CONS_SUPEQ: list(), ConsTyp.AP_CONS_SUP: list()}
    array = abstract1.to_lincons
    for i in range(len(array)):
        c = array.get(i)
        constraint = dict()
        cst = float(repr(c.get_cst()))
        if cst > 0:
            for name in inputs:
                constraint[name] = float(repr(c.get_coeff(PyVar(name)))) / cst
            constraint['_'] = float(repr(c.get_cst())) / cst
        elif cst < 0:
            for name in inputs:
                constraint[name] = float(repr(c.get_coeff(PyVar(name)))) / (- cst)
            constraint['_'] = float(repr(c.get_cst())) / (- cst)
        polyhedra[c.get_typ()].append(constraint)
    return polyhedra


def is_redundant(flattened, constraint, ranges):
    _ranges = dict(ranges)
    current = dict(
        objective=dict(
            name=None,
            coefficients=[
                {"name": name, "value": value} for name, value in constraint.items() if name != '_'
            ]),
        constraints=[
            dict(
                sense=1,
                pi=None,
                constant=c['_'],
                name=None,
                coefficients=[
                    {"name": name, "value": value} for name, value in c.items() if name != '_'
                ],
            )
            for c in flattened
        ],
        variables=[dict(lowBound=l, upBound=u, cat="Continuous", varValue=None, dj=None, name=v) for v, (l, u) in _ranges.items()],
        parameters=dict(name="NoName", sense=1, status=0, sol_status=0),
        sos1=list(),
        sos2=list(),
    )
    var, problem = pulp.LpProblem.fromDict(current)
    problem.solve(PULP_CBC_CMD(msg=False))
    return pulp.value(problem.objective) + constraint['_'] <= 0


def simplify_polyhedra(polyhedra, ranges):
    constraints = list(itertools.chain(polyhedra[ConsTyp.AP_CONS_SUPEQ], polyhedra[ConsTyp.AP_CONS_SUP]))
    mask = [True for _ in range(len(constraints))]
    for i, constraint in enumerate(constraints):
        flattened = [c for j, c in enumerate(constraints) if mask[j] and j != i]
        if is_redundant(flattened, constraint, ranges):
            mask[i] = False
    simplified = {ConsTyp.AP_CONS_SUPEQ: list(), ConsTyp.AP_CONS_SUP: list()}
    for i, constraint in enumerate(constraints):
        if mask[i]:     # add
            if i < len(polyhedra[ConsTyp.AP_CONS_SUPEQ]):
                simplified[ConsTyp.AP_CONS_SUPEQ].append(constraint)
            else:
                simplified[ConsTyp.AP_CONS_SUP].append(constraint)
    return simplified


def to_pulp(polyhedra, ranges, log=False):
    _ranges = dict(ranges)
    _ranges["__dummy"] = (0, 0)
    current = dict(
        objective=dict(name=None, coefficients=[{"name": "__dummy", "value": 1}]),
        constraints=[
            dict(
                sense=1,
                pi=None,
                constant=constraint['_'],
                name=None,
                coefficients=[
                    {"name": name, "value": value} for name, value in constraint.items() if name != '_'
                ],
            )
            for constraint in itertools.chain(polyhedra[ConsTyp.AP_CONS_SUPEQ], polyhedra[ConsTyp.AP_CONS_SUP])
        ],
        variables=[dict(lowBound=l, upBound=u, cat="Continuous", varValue=None, dj=None, name=v) for v, (l, u) in _ranges.items()],
        parameters=dict(name="NoName", sense=1, status=0, sol_status=0),
        sos1=list(),
        sos2=list(),
    )
    var, problem = pulp.LpProblem.fromDict(current)
    problem.solve(PULP_CBC_CMD(msg=False))
    if problem.status == -1:
        return None
    else:
        if log:
            counterexample = dict()
            for var in problem.variables():
                if var.name in ranges:
                    counterexample[var.name] = var.varValue
            bounds = dict()
            for name in ranges:
                _current = deepcopy(current)
                _current['objective'] = {'name': 'OBJ', 'coefficients': [{'name': name, 'value': 1}]}
                _current['parameters'] = {'name': '', 'sense': 1, 'status': 1, 'sol_status': 1}    # min
                _, _problem = pulp.LpProblem.fromDict(_current)
                _problem.solve(PULP_CBC_CMD(msg=False))
                lower = pulp.value(_problem.objective)
                current_ = deepcopy(current)
                current_['objective'] = {'name': 'OBJ', 'coefficients': [{'name': name, 'value': 1}]}
                current_['parameters'] = {'name': '', 'sense': -1, 'status': 1, 'sol_status': 1}     # max
                _, problem_ = pulp.LpProblem.fromDict(current_)
                problem_.solve(PULP_CBC_CMD(msg=False))
                upper = pulp.value(problem_.objective)
                bounds[name] = (lower, upper)
            return polyhedra, problem, counterexample, bounds
        else:
            return polyhedra


def analyze_polyhedra(polyhedra, flattened, active, inactive):
    try:
        (lhs, rhs) = next(flattened)
        if lhs in active:
            current = polyhedra
            # replace lhs with rhs in current constraints
            for constraint in itertools.chain(current[ConsTyp.AP_CONS_SUPEQ], current[ConsTyp.AP_CONS_SUP]):
                if lhs in constraint:
                    coeff = constraint[lhs]
                    del constraint[lhs]
                    for var, val in rhs.items():
                        if var in constraint:
                            constraint[var] += coeff * val
                        else:
                            constraint[var] = coeff * val
            # add rhs >= 0
            current[ConsTyp.AP_CONS_SUPEQ].append(rhs)
            yield from analyze_polyhedra(current, flattened, active, inactive)
        elif lhs in inactive:
            current = polyhedra
            # replace lhs with 0 in current constraints
            for constraint in itertools.chain(current[ConsTyp.AP_CONS_SUPEQ], current[ConsTyp.AP_CONS_SUP]):
                if lhs in constraint:
                    del constraint[lhs]
            # add rhs < 0
            current[ConsTyp.AP_CONS_SUP].append({k: -v for k, v in rhs.items()})
            yield from analyze_polyhedra(current, flattened, active, inactive)
        else:
            current1, current2 = deepcopy(polyhedra), deepcopy(polyhedra)
            for constraint in itertools.chain(current1[ConsTyp.AP_CONS_SUPEQ], current1[ConsTyp.AP_CONS_SUP]):
                if lhs in constraint:
                    coeff = constraint[lhs]
                    del constraint[lhs]
                    for var, val in rhs.items():
                        if var in constraint:
                            constraint[var] += coeff * val
                        else:
                            constraint[var] = coeff * val
            current1[ConsTyp.AP_CONS_SUPEQ].append(rhs)
            for constraint in itertools.chain(current2[ConsTyp.AP_CONS_SUPEQ], current2[ConsTyp.AP_CONS_SUP]):
                if lhs in constraint:
                    del constraint[lhs]
            current2[ConsTyp.AP_CONS_SUP].append({k: -v for k, v in rhs.items()})
            flattened1, flattened2 = itertools.tee(flattened)
            yield from analyze_polyhedra(current1, flattened1, active, inactive)
            yield from analyze_polyhedra(current2, flattened2, active, inactive)
    except StopIteration:
        yield polyhedra
