import chess
import copy

class GameState():
    def __init__(self, board=chess.Board(), SCORE_WIN=1000):
        self.board = board
        self.SCORE_WIN = SCORE_WIN

    def isEnd(self):
        return self.board.is_game_over()

    def getAgent(self):
        return 0 if self.board.turn == chess.WHITE else 1

    def getBoard(self):
        return self.board

    def getLegalActions(self):
        return list(self.board.legal_moves)

    def generateSuccessor(self, action):
        newBoard = copy.deepcopy(self.board)
        newBoard.push(action)
        return GameState(newBoard)

    def getNumAgents(self):
        return 2

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
