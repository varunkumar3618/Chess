import os
import os.path as path
import sys
import argparse
from collections import defaultdict

from testsuite import EPDTestSuite
from agent import UCIChessAgent, MTDAgent, RandomAgent

import numpy as np

from features import ALL_FEATURES

ROOT_PATH = path.dirname(path.realpath(__file__))
WEIGHTS_DIR = path.join(ROOT_PATH, "training-weights")

def extractFeatureVector(board):
    vectors = []
    for feature in ALL_FEATURES:
        vector = feature.value(board).flatten()
        vectors.append(vector)
    featureVector = np.concatenate(vectors)
    return featureVector

def MTDConstructor():
    weightsName = sorted(os.listdir(WEIGHTS_DIR))[-1]
    print "Using weights", weightsName
    weights = np.load(path.join(WEIGHTS_DIR, weightsName))
    def score(board):
        value = np.dot(weights, extractFeatureVector(board))
        return value
    return MTDAgent(name="MTD", depth=3, score=score)

AGENT_CONSTRUCTORS = {
    "stockfish": lambda: UCIChessAgent(name="Stockfish", engineFile=path.join(ROOT_PATH, "engines", "stockfish"), engineMoveTime=500),
    "mtd": MTDConstructor,
    "random": lambda: RandomAgent(name="rando")
}

def main(args):
    agent = AGENT_CONSTRUCTORS[args.agent]()
    numRuns = args.n
    epdFiles = [path.join(ROOT_PATH, "tests", "STS", "STS{}.epd".format(i)) \
                for i in xrange(1, 14)]
    suite = EPDTestSuite(agent, epdFiles)
    correctCounts = defaultdict(int)
    for _ in xrange(numRuns):
        results = suite.run()
        for filename, movePairs in results.iteritems():
            correctCount = sum(1 for movePair in movePairs if movePair[0] == movePair[1])
            correctCounts[filename] += correctCount
    for filename, correctCount in correctCounts.iteritems():
        print "{}: {} correct across {} runs".format(filename, correctCount, numRuns)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test chess engines against standard test suites")
    parser.add_argument("--agent", type=str, required=True,
                        help="the name of the chess agent to test",
                        choices=AGENT_CONSTRUCTORS.keys())
    parser.add_argument("-n", type=int, required=True,
                        help="the number of times to run each test")
    argv = parser.parse_args()
    sys.exit(main(argv))
