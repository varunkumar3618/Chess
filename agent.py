import chess, chess.uci

class UCIChessAgent(object):
    def __init__(self, engineFile, engineMoveTime):
        self.engine = chess.uci.popen_engine(engineFile)
        self.engine.uci()
        self.engineMoveTime = engineMoveTime
    def beginGame(self):
        self.engine.ucinewgame()
    def getMove(self, board):
        self.engine.position(board)
        return self.engine.go(movetime=self.engineMoveTime)[0]
