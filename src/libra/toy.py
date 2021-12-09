"""
Toy Example
===========

:Author: Caterina Urban
"""
import faulthandler
faulthandler.enable()
from libra.engine.bias_analysis import BiasAnalysis, AbstractDomain

spec = 'libra/tests/toy.txt'
nn = 'libra/tests/toy.py'
domain = AbstractDomain.DEEPPOLY_NEURIFY
L = 0.25
U = 2
BiasAnalysis(spec, domain=domain, startL=L, startU=U).main(nn)
