import numpy as np
from util import RLAlgorithm
from collections import defaultdict

class TDLambda(RLAlgorithm):
    def __init__(self, features, decay, alpha=0.01):
        self.discount = 1
        self.alpha = alpha
        self.decay = decay
        self.features = features
        self.weights = [np.zeros(f.shape(), dtype='float32') for f in self.features]
        self.stateCounts = defaultdict(int)
        self.beginSession()

    def getAction(self, state):
        valuesAndActions = [(self.value(state.generateSuccessor(action)), action) \
                            for action in state.getLegalActions()]
        return max(valuesAndActions)[1]

    def value(self, state):
        val = 0.
        for f, w in zip(self.features, self.weights):
            val += w.dot(f.value(state))
        return val

    def backup_value(self, state, scale):
        for f, w in zip(self.features, self.weights):
            w += scale * f.value(state)

    def beginSession(self):
        self.Z = defaultdict(float)

    def incorporateFeedback(self, state, action, reward, newState):
        futureValue = 0 if newState is None else self.discount * self.value(newState)
        delta = reward + futureValue - self.value(state)
        self.Z[state] += 1
        for s in self.Z:
            self.backup_value(s, self.alpha * delta * self.Z[s])
            self.Z[s] *= self.discount * self.decay
        self.stateCounts[state] += 1

    def getValues(self, states):
        return [self.value(s) for s in states]