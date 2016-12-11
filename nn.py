import numpy as np

"""
A very simple framwework for building a neural net. Keep in mind the following:
-> The Sequential layer uses the sub-layer names to disambiguate its parameters.
If you pass in multiple layers of the same type, you should be careful to name them
differently.
-> The parameter derivatives returned from the backward pass are in the same order
as the names returned by params.
"""

class Layer(object):
    def forward(self, input_t):
        raise NotImplementedError("Forward pass.")
    def backward(self, deriv, state):
        raise NotImplementedError("Backward pass.")
    def params(self):
        raise NotImplementedError("Parameter names.")
    def param_value(self, name):
        raise NotImplementedError("Parameter value.")
    def param_shape(self, name):
        raise NotImplementedError("Parameter shape.")
    def update_param(self, name, update):
        raise NotImplementedError("Perform parameter update.")

class Linear(Layer):
    def __init__(self, initial_value, name="Linear"):
        self.name = name
        self._W = np.copy(initial_value)
        self._param_deriv = None
    def forward(self, input_t):
        if len(input_t.shape) == 1:
            input_t = input_t[np.newaxis]
        return np.dot(input_t, self._W), [input_t]
    def backward(self, deriv, state):
        input_t, = state
        return np.dot(deriv, self._W.T), [np.dot(input_t.T, deriv)]
    def params(self):
        return ["W"]
    def param_shape(self, name):
        if name == "W":
            return self._W.shape
        else:
            raise ValueError("Invalid name.")
    def param_value(self, name):
        if name == "W":
            return self._W
        else:
            raise ValueError("Invalid name.")
    def update_param(self, name, update):
        if name == "W":
            self._W += update
        else:
            raise ValueError("Invalid name.")

class Bias(Layer):
    def __init__(self, initial_value, name="Bias"):
        self.name = name
        self._b = np.copy(initial_value)
    def forward(self, input_t):
        return input_t + self._b, None
    def backward(self, deriv, state):
        return deriv, [deriv]
    def params(self):
        return ["b"]
    def param_shape(self, name):
        if name == "b":
            return self._b.shape
        else:
            raise ValueError("Invalid name.")
    def param_value(self, name):
        if name == "b":
            return self._b
        else:
            raise ValueError("Invalid name.")
    def update_param(self, name, update):
        if name == "b":
            self._b += update
        else:
            raise ValueError("Invalid name.")

class Activation(Layer):
    def params(self):
        return []

class Sigmoid(Activation):
    def __init__(self, name="Sigmoid"):
        self.name = name
    def forward(self, input_t):
        output = 1 / (1 + np.exp(-input_t))
        return output, [output]
    def backward(self, deriv, state):
        output, = state
        return deriv * output * (1 - output), []

class Sequential(Layer):
    def __init__(self, layers, name="Sequential"):
        self.name = name
        self.layers = layers

        self.params_dict = {}
        self.params_list = []
        for layer in layers:
            for param in layer.params():
                full_name = layer.name + "/" + param
                self.params_dict[full_name] = (layer, param)
                self.params_list.append(full_name)
    def forward(self, input_t):
        states = []
        for layer in self.layers:
            input_t, state = layer.forward(input_t)
            states.append(state)
        return input_t, states
    def backward(self, deriv, states):
        all_params_deriv = []
        for layer, state in reversed(zip(self.layers, states)):
            deriv, params_deriv = layer.backward(deriv, state)
            all_params_deriv += params_deriv
        return deriv, all_params_deriv
    def params(self):
        return self.params_dict.keys()
    def param_shape(self, name):
        layer, param = self.params_dict[name]
        return layer.param_shape(param)
    def param_value(self, name):
        layer, param = self.params_dict[name]
        return layer.param_value(param)
    def update_param(self, name, update):
        layer, param = self.params_dict[name]
        layer.update_param(param, update)
