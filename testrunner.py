import os.path as path
import sys
import argparse

from testsuite import EPDTestSuite
from agent import UCIChessAgent, TDLambdaAgent

from features import Counts, PawnOccupation
from search import MinimaxSearchWithAlphaBeta

ROOT_PATH = path.dirname(path.realpath(__file__))

AGENT_CONSTRUCTORS = {
    "stockfish": lambda: UCIChessAgent(path.join(ROOT_PATH, "engines", "stockfish"), engineMoveTime=1),
    "tdlambda": lambda: TDLambdaAgent(MinimaxSearchWithAlphaBeta(), [Counts(), PawnOccupation()], 0.7, depth=2)
}

def main(args):
    agent = AGENT_CONSTRUCTORS[args.agent]()
    epdFiles = [path.join(ROOT_PATH, "tests", "STS", "STS{}.epd".format(i)) \
                for i in xrange(1, 14)]
    suite = EPDTestSuite(agent, epdFiles)
    results = suite.run()
    for filename, movePairs in results.iteritems():
        print "Results for ", filename
        correctCount = sum(1 for movePair in movePairs if movePair[0] == movePair[1])
        totalCount = len(movePairs)
        print "{}% correct".format(float(correctCount) / totalCount * 100)
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test chess engines against standard test suites")
    parser.add_argument("--agent", type=str, required=True,
                        help="the name of the chess agent to test",
                        choices=AGENT_CONSTRUCTORS.keys())
    argv = parser.parse_args()
    sys.exit(main(argv))
