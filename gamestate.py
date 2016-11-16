import chess

class TwoPlayerGameState(object):
    def isEnd(self):
        raise NotImplementedError("Bool.")

    def getAgentNo(self):
        raise NotImplementedError("Agent.")

    def getActions(self):
        raise NotImplementedError("Actions.")

    def generateSuccessor(self, action):
        raise NotImplementedError("TwoPlayerGameState")

    def getScore(self):
        raise NotImplementedError("Int.")

class ChessGameState(TwoPlayerGameState):
    def __init__(self, board=None, SCORE_WIN=1000):
        if board is None:
            board = chess.Board()
        self.board = board
        self.SCORE_WIN = SCORE_WIN

    def isEnd(self):
        return self.board.is_game_over()

    def getAgentNo(self):
        return self.board.turn

    def getBoard(self):
        return self.board

    def getActions(self):
        return list(self.board.legal_moves)

    def generateSuccessor(self, action):
        newBoard = chess.Board(fen=self.board.fen())
        newBoard.push(action)
        return ChessGameState(newBoard)

    def getScore(self):
        """
        Return the utility of the current board position for the white player
        """
        result = self.board.result()
        if self.board.is_game_over():
            if (result == "1-0"):
                return self.SCORE_WIN
            elif (result == "0-1"):
                return -self.SCORE_WIN
            else:
                return 0
        else:
            return 0

    def __hash__(self):
        return hash('%s, %s' % (self.board.fen(), self.SCORE_WIN))

    def __eq__(self, other):
        return isinstance(other, ChessGameState)\
            and self.board.fen() == other.board.fen()\
            and self.SCORE_WIN == other.SCORE_WIN
