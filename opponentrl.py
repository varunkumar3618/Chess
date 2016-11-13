from util import RLAlgorithm
import chess
import chess.uci

class OpponentRL(RLAlgorithm):
    """
    It never learns.
    """
    def __init__(self, engineFile, engineMoveTime):
        self.engine = chess.uci.popen_engine(engineFile)
        self.engineMoveTime = engineMoveTime

    def getAction(self, state):
        self.engine.position(state.getBoard())
        return self.engine.go(movetime=self.engineMoveTime)[0]

    def beginSession(self):
        pass

    def incorporateFeedback(self, state, action, reward, newState):
        pass
