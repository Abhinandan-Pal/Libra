from enum import Enum
from deeppoly_domain import init_deeppoly, analyze_deeppoly


class AbstractDomain(Enum):
    SYMBOLIC = 1
    DEEPPOLY = 2
    NEURIFY = 3
    PRODUCT = 4


def init(ranges):
    import config
    if config.domain == AbstractDomain.SYMBOLIC:
        return init_symbolic(ranges)
    elif config.domain == AbstractDomain.DEEPPOLY:
        return init_deeppoly(ranges)
    elif config.domain == AbstractDomain.NEURIFY:
        return init_neurify(ranges)
    else:
        assert config.domain == AbstractDomain.PRODUCT
        return init_product(ranges)


def analyze(state, inputs, layers, outputs):
    import config
    if config.domain == AbstractDomain.SYMBOLIC:
        return analyze_symbolic(state, inputs, layers, outputs)
    elif config.domain == AbstractDomain.DEEPPOLY:
        return analyze_deeppoly(state, inputs, layers, outputs)
    elif config.domain == AbstractDomain.NEURIFY:
        return analyze_neurify(state, inputs, layers, outputs)
    else:
        assert config.domain == AbstractDomain.PRODUCT
        return analyze_product(state, inputs, layers, outputs)
