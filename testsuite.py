import chess

from gamestate import ChessGameState

class TestSuite(object):
    def __init__(self, agent):
        self.agent = agent

    def run(self):
        # return a dictionary of {suitename: list of (move, bestMove) pairs}
        raise NotImplementedError("Override me!")

class EPDTestSuite(TestSuite):
    """
    Run tests specified by EPD (Extended Position Description) files.
    See https://gist.github.com/niklasf/73c9565719d124af64ff for an example
    with python-chess. For more on the format, see
    http://chessdb.sourceforge.net/tutorial/help/EPD.html.
    """
    def __init__(self, agent, filenames):
        super(EPDTestSuite, self).__init__(agent)
        # epdFiles is a dictionary of {filename: [lines in file]}
        self.epdFiles = {}
        for filename in filenames:
            with open(filename, "r") as epdFile:
                self.epdFiles[filename] = epdFile.read().splitlines()

    def showMoveResult(self, testNumber, move, bestMove):
        print "Test ", testNumber
        if move == bestMove:
            print "Correct! ({})".format(bestMove)
        else:
            print "Best:\t", bestMove
            print "Chose:\t", move
        print "--------------------"

    def testEPD(self, epdLine):
        position = chess.Board()
        epdInfo = position.set_epd(epdLine)
        bestMove = epdInfo["bm"][0]
        self.agent.beginEpisode(None, None) # FIXME
        move = self.agent.getAction(ChessGameState(position))
        return (move, bestMove)

    def run(self):
        # `results` is a dictionary of {filename: list of (move, bestMove) pairs}
        results = {}
        for filename, lines in sorted(self.epdFiles.iteritems()):
            print "Running tests from ", filename
            movePairs = []
            for testNumber, epdLine in enumerate(lines):
                move, bestMove = self.testEPD(epdLine)
                self.showMoveResult(testNumber + 1, move, bestMove)
                movePairs.append((move, bestMove))
            results[filename] = movePairs
        return results
