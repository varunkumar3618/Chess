import chess
import numpy as np

class TDAlgorithm(object):
    def beginEpisode(self, initialState):
        pass
    def incorporateFeedback(self, reward, newState):
        pass

class TDLambda(TDAlgorithm):
    def __init__(self, featureExtractor, decay=0.75, discount=1., learningRate=0.001, initialWeights=None):
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
        if reward == 0:
            newValue = np.dot(self.weights, newFeatures) # V'
        else:
            newValue = reward

        self.e = self.discount * self.decay * self.e + (
            1 - self.learningRate * self.discount * self.decay * np.dot(self.e, self.previousFeatures)
        ) * self.previousFeatures

        delta = self.discount * newValue - currentValue
        valueChange = currentValue - self.previousValue # V - Vold
        self.weights += self.learningRate * (delta + valueChange) * self.e \
                        - self.learningRate * valueChange * self.previousFeatures
        print "newValue", newValue
        print "currentValue", currentValue
        print "valueChange", valueChange
        self.previousValue = newValue
        self.previousFeatures = newFeatures

    def getWeights(self):
        return self.weights
