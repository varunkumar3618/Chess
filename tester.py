import numpy as np
import os
from os import path

from simulator import Simulator
from agent import MTDAgent
from features import ALL_FEATURES

def extractFeatureVector(board):
    vectors = []
    for feature in ALL_FEATURES:
        vector = feature.value(board).flatten()
        vectors.append(vector)
    featureVector = np.concatenate(vectors)
    return featureVector

WEIGHTS_DIR = "./training-weights"

def scoringFunction(weights):
    def score(board):
        value = np.dot(weights, extractFeatureVector(board))
        # print value
        return value
    return score

def main():
    previousWeights = None
    latestWeights = None
    savedWeights = sorted(os.listdir(WEIGHTS_DIR))
    if len(savedWeights) > 1:
        latestWeightsName = savedWeights[-1]
        print "Latest ", latestWeightsName
        latestWeights = np.load(path.join(WEIGHTS_DIR, latestWeightsName))
        print latestWeights
        previousWeightsName = savedWeights[0]
        print "Previous ", previousWeightsName
        previousWeights = np.load(path.join(WEIGHTS_DIR, previousWeightsName))
        print previousWeights
    sim = Simulator(verbose=True)
    numGames = 2
    strongAgent = MTDAgent(name="Strong MTD", depth=3, score=scoringFunction(latestWeights))
    weakAgent = MTDAgent(name="Weak MTD", depth=3, score=scoringFunction(previousWeights))
    winners = sim.simulate(playerAgent=strongAgent, opponentAgent=weakAgent, numGames=numGames)
    print "Winners:", winners

if __name__ == "__main__":
    main()
