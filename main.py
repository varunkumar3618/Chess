from simulator import simulate
from tdlearning import TDLambda
from opponentrl import OpponentRL
from gamestate import GameState

protagonistRL = TDLambda(decay=0.7)
opponentRL = OpponentRL(engineFile="./engines/stockfish", engineMoveTime=30)

print simulate(GameState(), protagonistRL, protagonistRL, maxMoveCount=30, verbose=True)
# for state, value in protagonistRL.getValues().iteritems():
#     print state.getBoard()
#     print value, "\n\n"
