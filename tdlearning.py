from collections import defaultdict
import numpy as np
import chess

from mtdagent import MTDAgent, UCIChessAgent, renderBoard
from features import ALL_FEATURES

MATE_VALUE = 60000 + 8*2700

class TDAlgorithm(object):
    def beginEpisode(self):
        pass
    def incorporateFeedback(self, state, reward, newState):
        pass
    def resetUpdates(self):
        pass
    def updates(self):
        pass

class TDLambda(TDAlgorithm):
    def __init__(self, decay, valueFunction, keyFunction, discount=1., alpha=0.01):
        self.decay = decay
        self.valueFunction = valueFunction
        self.key = keyFunction
        self.discount = discount
        self.alpha = alpha
        self.Z = defaultdict(float)
        self._updates = defaultdict(float)

    def resetUpdates(self):
        self._updates.clear()

    def beginEpisode(self):
        self.Z.clear()

    def incorporateFeedback(self, state, reward, newState):
        futureValue = self.discount * self.valueFunction(newState)
        delta = reward + futureValue - self.valueFunction(state)

        self.Z[self.key(state)] += 1
        for s in self.Z:
            self._updates[s] += self.alpha * delta * self.Z[s]
            self.Z[s] *= self.discount * self.decay

    def updates(self):
        return self._updates

class TDLeaf(TDAlgorithm):
    def __init__(self, decay, valueFunction, principalVariationFunction, keyFunction, discount=1., alpha=0.01):
        self.principalVariationFunction = principalVariationFunction
        self.tdLambda = TDLambda(decay, valueFunction, discount=discount, keyFunction=keyFunction, alpha=alpha)

    def beginEpisode(self):
        self.tdLambda.beginEpisode()

    def incorporateFeedback(self, state, reward, newState):
        self.tdLambda.incorporateFeedback(self.principalVariationFunction(state), reward, self.principalVariationFunction(newState))

    def resetUpdates(self):
        self.tdLambda.resetUpdates()

    def updates(self):
        return self.tdLambda.updates()


def tdlambda(score, blackAgent, depth, numGames, decay, discount, alpha, checkmateReward):
    alg = TDLambda(decay=decay, valueFunction=score, keyFunction=lambda b: b.epd(), discount=discount, alpha=alpha)
    whiteAgent = MTDAgent(depth, score)

    for game in range(numGames):
        board = chess.Board()
        nextBoard = chess.Board()
        whiteAgent.beginGame()
        blackAgent.beginGame()
        alg.beginEpisode()

        while not board.is_game_over():
            if board.turn == chess.WHITE:
                move = whiteAgent.getMove(board)
            else:
                move = blackAgent.getMove(board)

            nextBoard.push(move)
            renderBoard(board)
            print "\n"
            reward = 0.
            if board.is_game_over():
                if nextBoard.result() == "1-0":
                    reward = checkmateReward
                elif nextBoard.result() == "0-1":
                    reward = -checkmateReward
            alg.incorporateFeedback(board, reward, nextBoard)
            board.push(move)

    return alg.updates()

def main():
    weights = np.load("./train/2016-12-03 22:01:32.634428 pretrained weights.npy")
    def extractFeatureVector(board):
        vectors = []
        for feature in ALL_FEATURES:
            vector = feature.value(board).flatten()
            vectors.append(vector)
        featureVector = np.concatenate(vectors)
        return featureVector

    def score(board):
        return np.dot(weights, extractFeatureVector(board))
    # whiteAgent = MTDAgent(depth=4, score=evaluate)
    updates = tdlambda(
        score=score,
        blackAgent=UCIChessAgent('./engines/stockfish', 10),
        depth=1,
        numGames=1,
        decay=1.,
        discount=1.,
        alpha=0.01,
        checkmateReward=MATE_VALUE
    )
    print updates

if __name__ == '__main__':
    main()
