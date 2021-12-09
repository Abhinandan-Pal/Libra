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
domain = AbstractDomain.DEEPPOLY
L = 1
U = 6
BiasAnalysis(spec, domain=domain, startL=L, startU=U).main(nn)
