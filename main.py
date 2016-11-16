import chess
from agent import UCIChessAgent, TDLambdaAgent
from features import Counts, PawnOccupation
from search import MinimaxSearch, MinimaxSearchWithAlphaBeta
from gamestate import ChessGameState

def playGame(whiteAgent, blackAgent, maxMoveCount=100, numTrials=50, verbose=True, board=None):
    state = ChessGameState(board=board)
    sequence = [(None, 0, state)]
    whiteAgent.beginEpisode(state, chess.WHITE)
    blackAgent.beginEpisode(state, chess.BLACK)

    print state.getBoard()
    print '-------------------------------------------'
    moveNo = 0
    while moveNo < maxMoveCount and not state.isEnd():
        moveNo += 1

        if state.getAgentNo() == chess.WHITE:
            action = whiteAgent.getAction(state)
        else:
            action = blackAgent.getAction(state)
        successor = state.generateSuccessor(action)
        print action
        print successor.getBoard()
        print '-----------------------------------------'

        whiteReward = successor.getScore()
        whiteAgent.incorporateFeedback(state, action, whiteReward, successor)
        blackAgent.incorporateFeedback(state, action, -whiteReward, successor)

        sequence.append((action, whiteReward, successor))
        state = successor
    if verbose:
        for t in sequence:
            print 'Action: %s, white reward: %s, state: %s' % t

if __name__ == '__main__':
    protagonistAgent = TDLambdaAgent(MinimaxSearchWithAlphaBeta(), [Counts(), PawnOccupation()], 0.7, depth=2)
    antagonistAgent = UCIChessAgent(engineFile="./engines/stockfish", engineMoveTime=30)

    playGame(protagonistAgent, antagonistAgent, maxMoveCount=100, numTrials=50, verbose=False)