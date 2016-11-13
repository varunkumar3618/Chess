from simulator import simulate
from tdlearning import TDLambda
from opponentrl import OpponentRL
from gamestate import GameState
from features import Counts

protagonistRL = TDLambda([Counts()], decay=0.7)
opponentRL = OpponentRL(engineFile="./engines/stockfish", engineMoveTime=30)

print simulate(GameState(), protagonistRL, opponentRL, maxMoveCount=100, numTrials=50, verbose=False)
print protagonistRL.weights
