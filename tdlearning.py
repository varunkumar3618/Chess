from collections import defaultdict

class TDAlgorithm(object):
    def beginEpisode(self):
        pass
    def incorporateFeedback(self, state, reward, newState):
        pass

class TDLambda(TDAlgorithm):
    def __init__(self, decay, valueFunction, backupFunction, discount=1., alpha=0.01):
        self.decay = decay
        self.valueFunction = valueFunction
        self.backupFunction = backupFunction
        self.discount = discount
        self.alpha = alpha
        self.Z = defaultdict(float)

    def beginEpisode(self):
        self.Z.clear()

    def incorporateFeedback(self, state, reward, newState):
        futureValue = self.discount * self.valueFunction(newState)
        delta = reward + futureValue - self.valueFunction(newState)

        for s in self.Z:
            self.backupFunction(state, self.alpha * delta * self.Z[s])
            self.Z[s] *= self.discount * self.decay

class TDLeaf(TDAlgorithm):
    def __init__(self, decay, valueFunction, backupFunction, principalVariationFunction, discount=1., alpha=0.01):
        self.principalVariationFunction = principalVariationFunction
        self.tdLambda = TDLambda(decay, valueFunction, backupFunction, discount=discount, alpha=alpha)

    def beginEpisode(self):
        self.tdLambda.beginEpisode()

    def incorporateFeedback(self, state, reward, newState):
        self.tdLambda.incorporateFeedback(self.principalVariationFunction(state), reward, self.principalVariationFunction(newState))
