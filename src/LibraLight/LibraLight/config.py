from enum import Enum
from multiprocessing import cpu_count
from typing import List, Dict, Tuple
from colorama import Fore, Back
from abstract_domain import AbstractDomain


class SplittingHeuristic(Enum):
    RELU_POLARITY = 1
    OUTPUT_POLARITY = 2


""" Analysis """
domain: AbstractDomain = AbstractDomain.DEEPPOLY
min_difference = None
start_difference = None
start_unstable = None
max_unstable = None
cpu = 1 #cpu_count()
splitting = SplittingHeuristic.OUTPUT_POLARITY

""" Neural Network """
threshold = None
inputs: List[str] = list()
activations: List[str] = list()
layers: List[Dict[str, Dict[str, float]]] = list()
outputs = (str, str)

""" Input Specification """
sensitive: str = ""
values = Tuple[Tuple[float, float], Tuple[float, float]]
continuous = None
binary = None
bounds = Dict[str, Tuple[float, float]]

""" Pretty Printing """
colors = [
    Fore.LIGHTMAGENTA_EX,
    Back.BLACK + Fore.WHITE,
    Back.LIGHTRED_EX + Fore.BLACK,
    Back.MAGENTA + Fore.BLACK,
    Back.BLUE + Fore.BLACK,
    Back.CYAN + Fore.BLACK,
    Back.LIGHTGREEN_EX + Fore.BLACK,
    Back.YELLOW + Fore.BLACK,
]
