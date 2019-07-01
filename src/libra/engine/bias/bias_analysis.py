"""
Bias Analysis
=============

:Author: Caterina Urban
"""
import ast
import ctypes
import itertools
import os
import sys
import time
from abc import ABCMeta
from copy import deepcopy
from itertools import product
from queue import Queue
from typing import Optional, Tuple, Set, Dict, List

from apronpy.manager import FunId
from apronpy.polka import PyPolkaMPQstrict

from libra.abstract_domains.bias.bias_domain import BiasState
from libra.abstract_domains.numerical.interval_domain import BoxState, IntEnum
from libra.abstract_domains.numerical.octagon_domain import OctagonState
from libra.abstract_domains.numerical.polyhedra_domain import PolyhedraState
from libra.abstract_domains.numerical.taylor1p_domain import Taylor1pState
from libra.abstract_domains.state import State
from libra.core.cfg import Node, Function, Activation
from libra.core.expressions import BinaryComparisonOperation, Literal, VariableIdentifier, BinaryBooleanOperation
from libra.core.statements import Call, VariableAccess, Assignment
from libra.engine.backward import BackwardInterpreter
from libra.engine.forward import ForwardInterpreter
from libra.engine.result import AnalysisResult
from libra.engine.runner import Runner
from libra.frontend.cfg_generator import ast_to_cfg
from libra.semantics.backward import DefaultBackwardSemantics
from libra.semantics.forward import DefaultForwardSemantics
from libra.visualization.graph_renderer import CFGRenderer


class ActivationPatternInterpreter(ForwardInterpreter):

    def analyze(self, initial: BoxState):
        worklist = Queue()
        worklist.put(self.cfg.in_node)
        state = deepcopy(initial)
        activations, activated, deactivated = set(), set(), set()
        while not worklist.empty():
            current: Node = worklist.get()  # retrieve the current node
            # execute block
            if isinstance(current, Function):
                state = self.semantics.semantics(current.stmts, state)
            elif isinstance(current, Activation):
                activations.add(current)
                active, inactive = deepcopy(state), deepcopy(state)
                # active path
                state1 = self.semantics.ReLU_call_semantics(current.stmts[0], active, True)
                # inactive path
                state2 = self.semantics.ReLU_call_semantics(current.stmts[0], inactive, False)
                if state1.is_bottom():
                    deactivated.add(current)
                elif state2.is_bottom():
                    activated.add(current)
                state = state1.join(state2)
            else:
                for stmt in reversed(current.stmts):
                    state = self.semantics.semantics(stmt, state)
            # update worklist
            for node in self.cfg.successors(current):
                worklist.put(node)
        return activations, activated, deactivated


class ActivationPatternForwardSemantics(DefaultForwardSemantics):

    def assume_call_semantics(self, stmt: Call, state: State) -> State:
        argument = self.semantics(stmt.arguments[0], state).result
        state.assume(argument)
        state.result = set()
        return state

    def list_semantics(self, stmt, state) -> State:
        lhss = [self.semantics(assignment.left, state).result for assignment in stmt]
        rhss = [self.semantics(assignment.right, state).result for assignment in stmt]
        return state.assign(lhss, rhss)

    def ReLU_call_semantics(self, stmt: Call, state: State, active: bool = True) -> State:
        assert len(stmt.arguments) == 1  # exactly one argument is expected
        argument = stmt.arguments[0]
        assert isinstance(argument, VariableAccess)
        left = argument.variable
        right = Literal('0')
        if active:  # assume h >= 0
            cond = {BinaryComparisonOperation(left, BinaryComparisonOperation.Operator.GtE, right)}
            return state.assume(cond)
        else:  # assign h = 0, assume h < 0
            cond = {BinaryComparisonOperation(left, BinaryComparisonOperation.Operator.Lt, right)}
            return state.assume(cond).assign({left}, {right})


class JoinHeuristics(IntEnum):
    NotTop = 0


class BiasInterpreter(BackwardInterpreter):

    def __init__(self, cfg, semantics, widening, precursory=None):
        super().__init__(cfg, semantics, widening, precursory)
        self.sensitive = None           # sensitive feature
        self.uncontroversial1 = None    # uncontroversial features / one-hot encoded
        self.uncontroversial2 = None    # uncontroversial features / unary

        self.outputs = None             # output classes

        self.activations = None         # number of activation nodes
        self.active = None              # always active activations
        self.inactive = None            # always inactive activations
        self.heuristic = None           # join heuristic

    def pick(self, initial, values, ranges, pivot):
        # bound the sensitive feature between 0 and 1
        left = BinaryComparisonOperation(Literal('0'), BinaryComparisonOperation.Operator.LtE, self.sensitive[0])
        right = BinaryComparisonOperation(self.sensitive[0], BinaryComparisonOperation.Operator.LtE, Literal('1'))
        range = BinaryBooleanOperation(left, BinaryBooleanOperation.Operator.And, right)
        for sensitive in self.sensitive[1:]:
            left = BinaryComparisonOperation(Literal('0'), BinaryComparisonOperation.Operator.LtE, sensitive)
            right = BinaryComparisonOperation(sensitive, BinaryComparisonOperation.Operator.LtE, Literal('1'))
            conj = BinaryBooleanOperation(left, BinaryBooleanOperation.Operator.And, right)
            range = BinaryBooleanOperation(range, BinaryBooleanOperation.Operator.And, conj)
        # take into account lower and upper bound of all the other (uncontroversial) features
        for feature, (lower, upper) in ranges.items():
            left = BinaryComparisonOperation(Literal(str(lower)), BinaryComparisonOperation.Operator.LtE, feature)
            right = BinaryComparisonOperation(feature, BinaryComparisonOperation.Operator.LtE, Literal(str(upper)))
            conj = BinaryBooleanOperation(left, BinaryBooleanOperation.Operator.And, right)
            range = BinaryBooleanOperation(range, BinaryBooleanOperation.Operator.And, conj)
        print('\n---------------------------')
        print('Range: {}'.format(
            ', '.join('{} ∈ [{}, {}]'.format(feature, lower, upper) for feature, (lower, upper) in ranges.items())
        ))
        print('---------------------------\n')
        entry = deepcopy(initial.precursory).assume({range}) if range else initial.precursory
        # find the (abstract) activation patterns corresponding to each possible value of the sensitive feature
        feasible = True
        patterns = set()
        for value in values:
            result = deepcopy(entry).assume({value[1]})
            activations, active, inactive = self.precursory.analyze(result)
            disjunctions = len(activations) - len(active) - len(inactive)
            if disjunctions > self.widening:
                feasible = False
                break
            patterns.add((value, frozenset(activations), frozenset(active), frozenset(inactive)))
        # perform the analysis, if feasible, or partition the space of values of all the uncontroversial features
        if feasible:
            for ((literal, value), self.activations, self.active, self.inactive) in patterns:
                print(literal)
                print('activations: {{{}}}'.format(', '.join('{}'.format(activation) for activation in self.activations)))
                print('active: {{{}}}'.format(', '.join('{}'.format(active) for active in self.active)))
                print('inactive: {{{}}}'.format(', '.join('{}'.format(inactive) for inactive in self.inactive)))
                disjunctions = len(self.activations) - len(self.active) - len(self.inactive)
                paths = 2 ** disjunctions
                print('Paths: {}\n'.format(paths))
                # pick outcome
                for chosen in self.outputs:
                    print('--------- Outcome: {} ---------\n'.format(chosen))
                    remaining = self.outputs - {chosen}
                    discarded = remaining.pop()
                    outcome = BinaryComparisonOperation(discarded, BinaryComparisonOperation.Operator.Lt, chosen)
                    for discarded in remaining:
                        cond = BinaryComparisonOperation(discarded, BinaryComparisonOperation.Operator.Lt, chosen)
                        outcome = BinaryBooleanOperation(outcome, BinaryBooleanOperation.Operator.And, cond)
                    result = deepcopy(initial).assume({outcome}, bwd=True)
                    # run the bias analysis
                    start = time.time()
                    progress = 0
                    joined = deepcopy(result).bottom()
                    for state in self.proceed(self.cfg.out_node, deepcopy(result), list()):
                        progress += 1
                        if progress % 500 == 0:
                            print('\nProgress: {}/{}'.format(progress, paths), 'Time: {}s'.format(time.time() - start))
                        if state:
                            state = state.assume({range}).assume({value})
                            representation = repr(state)
                            if not representation.startswith('-1.0 >= 0'):
                                joined = joined.join(state)
                                print(representation)
                                print('Time: {}s\n'.format(time.time() - start))
                    print('Joined: {}\n'.format(joined))
                print('---------\n')
        else:       # too many disjunctions, we need to split further
            print('Too many disjunctions ({})!'.format(disjunctions))
            (lower, upper) = ranges[self.uncontroversial2[pivot]]
            middle = lower + (upper - lower) / 2
            print('Split at: {}'.format(middle))
            left = deepcopy(ranges)
            left[self.uncontroversial2[pivot]] = (lower, middle)
            right = deepcopy(ranges)
            right[self.uncontroversial2[pivot]] = (middle, upper)
            self.pick(initial, values, left, (pivot + 1) % len(self.uncontroversial2))
            self.pick(initial, values, right, (pivot + 1) % len(self.uncontroversial2))

    def proceed(self, node, initial, path):
        # print('node: ', node)
        state = initial

        if isinstance(node, Function):
            state = self.semantics.semantics(node.stmts, state)
            if state.is_bottom():
                yield None
            else:
                predecessors = self.cfg.predecessors(node)
                if predecessors:
                    yield from self.proceed(self.cfg.predecessors(node).pop(), state, path)
                else:
                    yield state
        elif isinstance(node, Activation):
            if node in self.active:  # only the active path is viable
                state = self.semantics.ReLU_call_semantics(node.stmts[0], state, True)
                if state.is_bottom():
                    yield None
                else:
                    predecessor = self.cfg.predecessors(node).pop()
                    active_path = path + ['{}+{}'.format(node, predecessor)]
                    yield from self.proceed(predecessor, state, active_path)
            elif node in self.inactive:  # only the inactive path is viable
                state = self.semantics.ReLU_call_semantics(node.stmts[0], state, False)
                if state.is_bottom():
                    yield None
                else:
                    predecessor = self.cfg.predecessors(node).pop()
                    inactive_path = path + ['{}-{}'.format(node, predecessor)]
                    yield from self.proceed(predecessor, state, inactive_path)
            else:  # both paths are viable
                active, inactive = deepcopy(state), deepcopy(state)
                state1 = self.semantics.ReLU_call_semantics(node.stmts[0], active, True)
                state2 = self.semantics.ReLU_call_semantics(node.stmts[0], inactive, False)
                join = False
                if self.heuristic == JoinHeuristics.NotTop:
                    state = deepcopy(state1).join(state2)
                    if not state.is_top():
                        join = True
                if join:
                    predecessor = self.cfg.predecessors(node).pop()
                    join_path = path + ['{}*{}'.format(node, predecessor)]
                    yield from self.proceed(predecessor, state, join_path)
                else:
                    if state1.is_bottom():
                        if state2.is_bottom():
                            yield None
                        else:
                            predecessor = self.cfg.predecessors(node).pop()
                            inactive_path = path + ['{}-{}'.format(node, predecessor)]
                            yield from self.proceed(predecessor, state2, inactive_path)
                    else:
                        predecessor = self.cfg.predecessors(node).pop()
                        active_path = path + ['{}+{}'.format(node, predecessor)]
                        yield from self.proceed(predecessor, state1, active_path)
                        if state2.is_bottom():
                            yield None
                        else:
                            inactive_path = path + ['{}-{}'.format(node, predecessor)]
                            yield from self.proceed(predecessor, state2, inactive_path)
        else:
            stop = False
            for stmt in reversed(node.stmts):
                state = self.semantics.semantics(stmt, state)
                if state.is_bottom():
                    stop = True
                    break
            if stop:
                yield None
            else:
                predecessors = self.cfg.predecessors(node)
                if predecessors:
                    yield from self.proceed(self.cfg.predecessors(node).pop(), state, path)
                else:
                    yield state

    def one_hots(self, variables: List[VariableIdentifier]):
        """Compute all possible one-hots for a given list of variables.

        :param variables: list of variables one-hot encoding a categorical input feature
        :return: set of Libra expressions corresponding to each possible value of the one-hot encoding
        (paired with a string representing the encoded value for convenience ---
        the string is the first element of the tuple)
        """
        values = set()
        arity = len(variables)
        for i in range(arity):
            # the current variable has value one
            one = Literal('1')
            lower = BinaryComparisonOperation(one, BinaryComparisonOperation.Operator.LtE, variables[i])
            upper = BinaryComparisonOperation(variables[i], BinaryComparisonOperation.Operator.LtE, one)
            value = BinaryBooleanOperation(lower, BinaryBooleanOperation.Operator.And, upper)
            # everything else has value zero
            zero = Literal('0')
            for j in range(0, i):
                lower = BinaryComparisonOperation(zero, BinaryComparisonOperation.Operator.LtE, variables[j])
                upper = BinaryComparisonOperation(variables[j], BinaryComparisonOperation.Operator.LtE, zero)
                conj = BinaryBooleanOperation(lower, BinaryBooleanOperation.Operator.And, upper)
                value = BinaryBooleanOperation(conj, BinaryBooleanOperation.Operator.And, value)
            for j in range(i+1, arity):
                lower = BinaryComparisonOperation(zero, BinaryComparisonOperation.Operator.LtE, variables[j])
                upper = BinaryComparisonOperation(variables[j], BinaryComparisonOperation.Operator.LtE, zero)
                conj = BinaryBooleanOperation(lower, BinaryBooleanOperation.Operator.And, upper)
                value = BinaryBooleanOperation(value, BinaryBooleanOperation.Operator.And, conj)
            values.add(('{} = 1'.format(variables[i]), value))
        return values


    def analyze(self, initial: BiasState, inputs=None, outputs=None, heuristic: JoinHeuristics = None):
        # pick sensitive feature / we assume one-hot encoding
        arity = int(input('Arity of the sensitive feature?\n'))
        self.sensitive = list()
        for i in range(arity):
            self.sensitive.append(VariableIdentifier(input('Sensitive input:\n')))
        values = self.one_hots(self.sensitive)
        # determine the one-hot encoded uncontroversial features
        self.uncontroversial1 = list()
        while True:
            try:
                arity = input('Arity of the feature?\n')
                uncontroversial = list()
                for i in range(int(arity)):
                    uncontroversial.append(VariableIdentifier(input('Input:\n')))
                self.uncontroversial1.append(uncontroversial)
            except EOFError:
                break
        # determine the other uncontroversial features
        self.uncontroversial2 = list(inputs - set(self.sensitive) - set(itertools.chain(*self.uncontroversial1)))
        # do the rest
        self.outputs = outputs
        ranges = dict()
        for uncontroversial in self.uncontroversial2:
            ranges[uncontroversial] = (0, 1)
        self.pick(initial, values, ranges, 0)
        print('Done!')


class BiasBackwardSemantics(DefaultBackwardSemantics):

    def assume_call_semantics(self, stmt: Call, state: State) -> State:
        argument = self.semantics(stmt.arguments[0], state).result
        state.assume(argument, bwd=True)
        state.result = set()
        return state

    def list_semantics(self, stmt, state) -> State:
        lhss = [self.semantics(assignment.left, state).result for assignment in stmt]
        rhss = [self.semantics(assignment.right, state).result for assignment in stmt]
        return state.substitute(lhss, rhss)

    def ReLU_call_semantics(self, stmt: Call, state: State, active: bool = True) -> State:
        assert len(stmt.arguments) == 1  # exactly one argument is expected
        argument = stmt.arguments[0]
        assert isinstance(argument, VariableAccess)
        left = argument.variable
        right = Literal('0')
        if active:  # assume h >= 0
            cond = {BinaryComparisonOperation(left, BinaryComparisonOperation.Operator.GtE, right)}
            return state.assume(cond)
        else:  # assign h = 0, assume h < 0
            cond = {BinaryComparisonOperation(left, BinaryComparisonOperation.Operator.Lt, right)}
            return state.substitute({left}, {right}).assume(cond)


class BiasAnalysis(Runner):

    def __init__(self):
        super().__init__()
        self.inputs = None
        self.outputs = None

    def interpreter(self):
        precursory = ActivationPatternInterpreter(self.cfg, ActivationPatternForwardSemantics(), 3)
        return BiasInterpreter(self.cfg, BiasBackwardSemantics(), 3, precursory=precursory)

    def state(self):
        self.inputs, variables, self.outputs = self.variables
        precursory = BoxState(variables)
        # precursory = OctagonState(self.variables)
        # precursory = PolyhedraState(self.variables)
        min_int = (-ctypes.c_uint(-1).value) // 2
        PyPolkaMPQstrict.manager.contents.option.funopt[FunId.AP_FUNID_IS_BOTTOM].algorithm = min_int
        PyPolkaMPQstrict.manager.contents.option.funopt[FunId.AP_FUNID_MEET].algorithm = min_int
        return BiasState(variables, precursory=precursory)

    @property
    def variables(self):
        variables, assigned, outputs = set(), set(), set()
        worklist = Queue()
        worklist.put(self.cfg.in_node)
        while not worklist.empty():
            current = worklist.get()
            for stmt in current.stmts:
                variables = variables.union(stmt.ids())
                if isinstance(stmt, Assignment):
                    assigned = assigned.union(stmt.left.ids())
                    outputs = outputs.union(stmt.left.ids())
            if isinstance(current, Activation):  # there is another layer
                outputs = set()
            for node in self.cfg.successors(current):
                worklist.put(node)
        return variables.difference(assigned), variables, outputs

    def main(self, path, heuristic: JoinHeuristics = None):
        self.path = path
        with open(self.path, 'r') as source:
            self.source = source.read()
            self.tree = ast.parse(self.source)
            self.cfg = ast_to_cfg(self.tree)
            # renderer = CFGRenderer()
            # data = self.cfg
            # name = os.path.splitext(os.path.basename(self.path))[0]
            # label = f"CFG for {name}"
            # directory = os.path.dirname(self.path)
            # renderer.render(data, filename=name, label=label, directory=directory, view=True)
        self.run(heuristic=heuristic)

    def run(self, heuristic: JoinHeuristics = None):
        start = time.time()
        self.interpreter().analyze(self.state(), inputs=self.inputs, outputs=self.outputs, heuristic=heuristic)
        end = time.time()
        print('Total: {}s'.format(end - start))
