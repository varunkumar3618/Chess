import numpy as np
import datetime
import os

from tdlearning import TDLambda
from agent import UCIChessAgent
from features import ALL_FEATURES
from simulator import Simulator

def extractFeatureVector(board):
    vectors = []
    for feature in ALL_FEATURES:
        vector = feature.value(board).flatten()
        vectors.append(vector)
    featureVector = np.concatenate(vectors)
    return featureVector

def main():
    initialWeights = None
    savedWeights = os.listdir("./training-weights/")
    if len(savedWeights) > 0:
        weightsName = sorted(savedWeights)[-1]
        print "Loading ", weightsName
        initialWeights = np.load("./training-weights/{}".format(weightsName))

    tdAlg = TDLambda(featureExtractor=extractFeatureVector, initialWeights=initialWeights)
    numGames=20
    strongAgent = UCIChessAgent("Strong Stockfish", './engines/stockfish', 50)
    weakAgent = UCIChessAgent("Weak Stockfish", './engines/stockfish', 10)
    sim = Simulator(verbose=True, tdAlgorithm=tdAlg)
    winners = sim.simulate(playerAgent=strongAgent, opponentAgent=weakAgent, numGames=numGames)
    print "Winners: ", winners
    weights = tdAlg.getWeights()
    print weights
    now = datetime.datetime.now()
    np.save("./training-weights/{}-{} games".format(now, numGames), weights)

if __name__ == "__main__":
    main()
