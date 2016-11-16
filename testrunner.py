import chess
import chess.uci
import os.path as path
import sys

from testsuite import EPDTestSuite
from enginecontainer import UCIEngineContainer

ROOT_PATH = path.dirname(path.realpath(__file__))

def main():
    engine = chess.uci.popen_engine(path.join(ROOT_PATH, "engines", "stockfish"))
    engineContainer = UCIEngineContainer(engine)
    epdFiles = [path.join(ROOT_PATH, "tests", "STS", "STS{}.epd".format(i)) \
                for i in xrange(1, 14)]
    suite = EPDTestSuite(engineContainer, epdFiles)
    results = suite.run(engineMoveTime=1)
    for filename, movePairs in results.iteritems():
        print "Results for ", filename
        correctCount = sum(1 for movePair in movePairs if movePair[0] == movePair[1])
        totalCount = len(movePairs)
        print "{}% correct".format(float(correctCount) / totalCount * 100)

if __name__ == "__main__":
    sys.exit(main())
