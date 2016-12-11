import numpy as np

import nn

"""
Source: https://web.stanford.edu/group/pdplab/pdphandbook/

-> The model should be conform to the interfaces in nn and have an output of a single unit.
"""

class TDNet(object):
    def __init__(self, feature_extractor, model, decay, learning_rate):
        self.feature_extractor = feature_extractor
        self.model = model
        self.decay = decay
        self.learning_rate = learning_rate
        self.traces = {}

    def begin_game(self):
        self.traces = {}
        for param in self.model.params():
            self.traces[param] = np.zeros(self.model.param_shape(param), dtype="float32")

    def evaluate(self, board):
        input_t = self.feature_extractor(board)
        return self.model.forward(input_t)[0]

    def update_trace(self, board):
        input_t = self.feature_extractor(board)
        _, state = self.model.forward(input_t)
        _, param_derivs = self.model.backward(1., state)
        for param, deriv in zip(self.model.params(), param_derivs):
            self.traces[param] = self.decay * self.traces[param] + deriv

    def update_params(self, td):
        for param in self.model.params():
            self.model.update_param(param, self.learning_rate * td * self.traces[param])
