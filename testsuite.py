import chess

class TestSuite(object):
    def __init__(self, engineContainer):
        self.engineContainer = engineContainer

    def run(self, engineMoveTime):
        # return a dictionary of {suitename: list of (move, bestMove) pairs}
        raise NotImplementedError("Override me!")

class EPDTestSuite(TestSuite):
    def __init__(self, engineContainer, filenames):
        super(EPDTestSuite, self).__init__(engineContainer)
        # epdFiles is a dictionary of {filename: [lines in file]}
        self.epdFiles = {}
        for filename in filenames:
            with open(filename, "r") as epdFile:
                self.epdFiles[filename] = epdFile.read().splitlines()

    def showMoveResult(self, testNumber, move, bestMove):
        print "Test ", testNumber
        if move == bestMove:
            print "Correct!"
        else:
            print "Best:\t", bestMove
            print "Chose:\t", move
        print "--------------------"

    def testEPD(self, epdLine, engineMoveTime):
        position = chess.Board()
        epdInfo = position.set_epd(epdLine)
        self.engineContainer.newGame()
        self.engineContainer.position(position)
        move = self.engineContainer.go(moveTime=engineMoveTime)[0]
        bestMove = epdInfo["bm"][0]
        return (move, bestMove)

    def run(self, engineMoveTime):
        # `results` is a dictionary of {filename: list of (move, bestMove) pairs}
        results = {}
        for filename, lines in sorted(self.epdFiles.iteritems()):
            print "Running tests from ", filename
            movePairs = []
            for testNumber, epdLine in enumerate(lines):
                move, bestMove = self.testEPD(epdLine, engineMoveTime)
                self.showMoveResult(testNumber + 1, move, bestMove)
                movePairs.append((move, bestMove))
            results[filename] = movePairs
        return results
