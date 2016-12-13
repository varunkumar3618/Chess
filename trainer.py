import numpy as np
import datetime
import os

from tdlearning import TDLambda
from agent import UCIChessAgent
from features import ALL_FEATURES, extractFeatureVector
from simulator import Simulator

def main():
    numBatches = 100
    for batchNum in xrange(numBatches):
        print "Batch {} of {}".format(batchNum + 1, numBatches)
        initialWeights = None
        savedWeights = os.listdir("./training-weights/")
        if len(savedWeights) > 0:
            weightsName = sorted(savedWeights)[-1]
            print "Loading ", weightsName
            initialWeights = np.load("./training-weights/{}".format(weightsName))

        tdAlg = TDLambda(featureExtractor=lambda b: extractFeatureVector(ALL_FEATURES, b), initialWeights=initialWeights)
        numGames = 20
        strongAgent = UCIChessAgent("Strong Stockfish", './engines/stockfish', 100)
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
