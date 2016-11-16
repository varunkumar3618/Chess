import numpy as np
from tdlearning import TDLambda
import chess.uci

class Agent(object):
    def beginEpisode(self, state, params):
        pass
    def getAction(self, state):
        raise NotImplementedError("Action.")
    def incorporateFeedback(self, state, action, reward, newState):
        pass
    def endEpisode(self):
        pass


class UCIChessAgent(Agent):
    def __init__(self, engineFile, engineMoveTime):
        self.engine = chess.uci.popen_engine(engineFile)
        self.engine.uci()
        self.engineMoveTime = engineMoveTime
    def beginEpisode(self, state, params):
        pass
    def getAction(self, state):
        self.engine.position(state.getBoard())
        return self.engine.go(movetime=self.engineMoveTime)[0]
    def incorporateFeedback(self, state, action, reward, newState):
        pass
    def endEpisode(self):
        pass


class TDLambdaAgent(Agent):
    def __init__(self, search, features, decay, depth=5, discount=1., alpha=0.01):
        self.agentNo = None
        self.search = search
        self.features = features
        self.weights = {}
        for feature in self.features:
            self.weights[feature] = np.random.normal(size=feature.shape).astype('float32')
        self.depth = depth

        def valueFunction(state):
            return self._computeValue(state)
        self.search.configure(valueFunction)

        def backupFunction(state, scale):
            self._backup(state, scale)

        self.tdLambda = TDLambda(decay, valueFunction, backupFunction, discount=discount, alpha=alpha)

    def beginEpisode(self, state, agentNo):
        self.agentNo = agentNo
        self.tdLambda.beginEpisode()

    def getAction(self, state):
        principalVariationPath = self.search.principalVariationPath(state, self.depth)
        return principalVariationPath[0][-1]

    def incorporateFeedback(self, state, action, reward, newState):
        self.tdLambda.incorporateFeedback(state, reward, newState)

    def _computeValue(self, state):
        value = sum(np.sum(weight * feature.value(state)) for feature, weight in self.weights.items())
        if state.getAgentNo() == self.agentNo:
            return value
        else:
            return -value

    def _backup(self, state, scale):
        for feature in self.features:
            self.weights[feature] += feature.value(state) * scale