import chess
import numpy as np

from tdpredictor import TDNet

class TDAlgorithm(object):
    def beginEpisode(self, initialState):
        pass
    def incorporateFeedback(self, reward, newState):
        pass

class TDLambda(TDAlgorithm):
    def __init__(self, decay, featureExtractor, discount=1., learningRate=0.01, initialWeights=None):
        self.decay = decay # lambda
        self.featureExtractor = featureExtractor # Phi
        self.discount = discount # gamma
        self.learningRate = learningRate # alpha
        sampleFeature = self.featureExtractor(chess.Board())
        self.e = np.zeros(sampleFeature.shape)
        if initialWeights is None:
            self.weights = np.zeros(sampleFeature.shape) # theta
        else:
            self.weights = initialWeights
        self.previousValue = 0 # Initialized in beginEpisode
        self.previousFeatures = None # Initialized in beginEpisode

    def beginEpisode(self, initialState):
        self.e *= 0
        self.previousValue = 0
        self.previousFeatures = self.featureExtractor(initialState)

    def incorporateFeedback(self, reward, newState):
        currentValue = np.dot(self.weights, self.previousFeatures) # V
        newFeatures = self.featureExtractor(newState) # Phi'
        newValue = np.dot(self.weights, newFeatures) # V'
        self.e = self.discount * self.decay * self.e + (
            1 - self.learningRate * self.discount * self.decay * np.dot(self.e, self.previousFeatures)
        ) * self.previousFeatures

        delta = reward + self.discount * newValue - currentValue
        valueChange = currentValue - self.previousValue # V - Vold
        self.weights += self.learningRate * (delta + valueChange) * self.e \
                        - self.learningRate * valueChange * self.previousFeatures

        self.previousValue = newValue
        self.previousFeatures = newFeatures

    def getWeights(self):
        return self.weights

class TDNetLambda(TDAlgorithm):
    def __init__(self, decay, featureExtractor, model, learningRate=0.01):
        self.net = TDNet(featureExtractor, model, decay, learningRate)
        self.previousState = self.previousValue = None

    def beginEpisode(self, initialState):
        self.net.begin_game()
        self.previousState = initialState
        self.previousValue = self.net.evaluate(initialState)

    def incorporateFeedback(self, reward, newState):
        if reward == 0:
            newValue = self.net.evaluate(newState)
        else:
            newValue = reward
        self.net.update_trace(self.previousState)
        self.net.update_params(newValue - self.previousValue)
        self.previousState = newState
        self.previousValue = newValue

    def getWeights(self):
        return np.zeros((1, 1))
