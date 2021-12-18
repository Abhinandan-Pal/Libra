
def parse_keras(model, distance):
    n_ins = model.layers[0].get_weights()[0].shape[0]
    inputs = ["x0%d" % j for j in range(0, n_ins)]
    activations = list()
    outputs = None
    #
    layers = list()
    l = 1
    for layer in model.layers:
        weights = layer.get_weights()[0]
        biases = layer.get_weights()[1]
        ins = layer.get_weights()[0].shape[0]
        outs = layer.get_weights()[0].shape[1]
        #
        current = dict()
        for i in range(0, outs):
            if layer.get_config().get('activation', None) == 'relu':
                lhs = "x%d%d" % (l, i)
                rhs = dict()
                for j in range(0, ins):
                    rhs["x%d%d" % (l - 1, j)] = weights[j][i]
                rhs["_"] = biases[i]
                activations.append(lhs)
                current[lhs] = rhs
            else:   # patching the output layer
                assert outs == 1
                lhs1 = "x%d%d" % (l, i)
                rhs1 = dict()
                for j in range(0, ins):
                    rhs1["x%d%d" % (l - 1, j)] = - weights[j][i]
                rhs1["_"] = distance - biases[i]
                current[lhs1] = rhs1
                lhs2 = "x%d%d" % (l, i + 1)
                rhs2 = dict()
                for j in range(0, ins):
                    rhs2["x%d%d" % (l - 1, j)] = weights[j][i]
                rhs2["_"] = biases[i] - distance
                current[lhs2] = rhs2
                outputs = (lhs1, lhs2)
        #
        layers.append(current)
        l = l + 1
    return inputs, activations, layers, outputs
