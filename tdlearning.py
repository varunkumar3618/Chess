from util import RLAlgorithm
from collections import defaultdict

class TDLambda(RLAlgorithm):
    def __init__(self, decay, alpha=0.7):
        self.discount = 1
        self.alpha = alpha
        self.decay = decay
        self.values = defaultdict(float)
        self.stateCounts = defaultdict(int)
        self.beginSession()

    def getAction(self, state):
        values = self.getValues()
        valuesAndActions = [(values[state.generateSuccessor(action)], action) \
                            for action in state.getLegalActions()]
        return max(valuesAndActions)[1]

    def beginSession(self):
        self.Z = defaultdict(float)

    def incorporateFeedback(self, state, action, reward, newState):
        futureValue = 0 if newState is None else self.discount * self.values[newState]
        delta = reward + futureValue - self.values[state]
        self.Z[state] += 1
        for s in self.Z:
            self.values[s] += self.alpha * delta * self.Z[s]
            self.Z[s] *= self.discount * self.decay
        self.stateCounts[state] += 1

    def getValues(self):
        return self.values
