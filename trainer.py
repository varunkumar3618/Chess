import chess
import numpy as np
import datetime

from tdlearning import TDLambda
from agent import UCIChessAgent
from util import renderBoard
from features import ALL_FEATURES

def extractFeatureVector(board):
    vectors = []
    for feature in ALL_FEATURES:
        vector = feature.value(board).flatten()
        vectors.append(vector)
    featureVector = np.concatenate(vectors)
    return featureVector

def simulate(tdAlg, numGames=10):
    playerAgent = UCIChessAgent('./engines/stockfish', 50)
    opponentAgent = UCIChessAgent('./engines/stockfish', 10)
    for game in xrange(numGames):
        if game % 2 == 0:
            print "Stronger agent playing as white"
            whiteAgent = playerAgent
            blackAgent = opponentAgent
        else:
            print "Stronger agent playing as black"
            whiteAgent = opponentAgent
            blackAgent = playerAgent
        print "Game {} of {}".format(game + 1, numGames)
        board = chess.Board()
        nextBoard = chess.Board()
        whiteAgent.beginGame()
        blackAgent.beginGame()
        tdAlg.beginEpisode(board)

        renderBoard(board); print "\n"

        while not board.is_game_over():
            if board.turn == chess.WHITE:
                move = whiteAgent.getMove(board)
            else:
                move = blackAgent.getMove(board)

            nextBoard.push(move)
            renderBoard(nextBoard); print "\n"
            reward = 0.
            if nextBoard.is_game_over():
                print "Game over", nextBoard.result()
                print "-----------------------------------------------"
                if nextBoard.result() == "1-0":
                    reward = 1
                elif nextBoard.result() == "0-1":
                    reward = 0
                else:
                    reward = 0.5
            tdAlg.incorporateFeedback(reward, nextBoard)
            board.push(move)
    return tdAlg.weights

def main():
    initialWeights = None
    # initialWeights = np.load("./training-weights/...")
    tdAlg = TDLambda(decay=0.75, featureExtractor=extractFeatureVector, initialWeights=initialWeights)
    simulate(tdAlg, numGames=10)
    weights = tdAlg.getWeights()
    now = datetime.datetime.now()
    np.save("./training-weights/{}".format(now), weights)

if __name__ == "__main__":
    main()
