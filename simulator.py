import chess
from util import renderBoard

class Simulator(object):
    def __init__(self, verbose=False, tdAlgorithm=None):
        self.verbose = verbose
        self.tdAlgorithm = tdAlgorithm

    def display(self, *toPrint):
        if self.verbose:
            print " ".join(map(str, toPrint))

    def beginEpisode(self, board):
        if self.tdAlgorithm:
            self.tdAlgorithm.beginEpisode(board)

    def incorporateFeedback(self, reward, nextBoard):
        if self.tdAlgorithm:
            self.tdAlgorithm.incorporateFeedback(reward, nextBoard)

    def simulate(self, playerAgent, opponentAgent, numGames=10):
        winners = []
        for game in xrange(numGames):
            if game % 2 == 0:
                whiteAgent = playerAgent
                blackAgent = opponentAgent
            else:
                whiteAgent = opponentAgent
                blackAgent = playerAgent

            self.display("Game {} of {}".format(game + 1, numGames))
            self.display("White: {}, Black: {}".format(whiteAgent.getName(), blackAgent.getName()))

            board = chess.Board()
            nextBoard = chess.Board()
            whiteAgent.beginGame()
            blackAgent.beginGame()

            self.beginEpisode(board)
            self.display(renderBoard(board), "\n")
            moveNumber = 1

            while not board.is_game_over():
                if board.turn == chess.WHITE:
                    move = whiteAgent.getMove(board)
                else:
                    move = blackAgent.getMove(board)

                nextBoard.push(move)
                self.display("Move ", moveNumber)
                self.display(renderBoard(nextBoard), "\n")

                reward = 0.
                if nextBoard.is_game_over():
                    self.display("Game over", nextBoard.result())
                    self.display("-----------------------------------------------")

                    if nextBoard.result() == "1-0":
                        reward = 1
                        winners.append(whiteAgent.getName())
                    elif nextBoard.result() == "0-1":
                        reward = -1
                        winners.append(blackAgent.getName())
                    else:
                        reward = 0
                        winners.append(None)

                self.incorporateFeedback(reward, nextBoard)
                board.push(move)
                moveNumber += 1
        return winners
