import sys

import chess

from mtdagent import TDMTDAgent, evaluate, renderBoard

class BenchmarkAgent(TDMTDAgent):
    def __init__(self, depth):
        super(BenchmarkAgent, self).__init__(depth=depth, features=[])

    def _score(self, board):
        return evaluate(board)

def simulate(whiteAgent, blackAgent, verbose=True):
    board = chess.Board()
    # board = chess.Board(fen="8/1pp4p/8/8/1Pb2nP1/p4Pk1/7r/2R4K w - - 4 33")
    whiteAgent.beginGame()
    blackAgent.beginGame()
    while not board.is_game_over():
        if verbose:
            renderBoard(board)
            # print board.fen()
        if board.turn == chess.WHITE:
            move = whiteAgent.getMove(board)
        else:
            move = blackAgent.getMove(board)
        if verbose:
            if not board.is_game_over():
                print move
            print '-----------------------------'
        board.push(move)
    if verbose:
        renderBoard(board)
        board.reset()
        print '================================'
    return board.result()

if __name__ == '__main__':
    depth1 = int(sys.argv[1])
    depth2 = int(sys.argv[2])
    num_games = int(sys.argv[3])

    whiteAgent = BenchmarkAgent(depth1)
    blackAgent = BenchmarkAgent(depth2)

    whiteWins = 0
    blackWins = 0
    draws = 0

    for game in range(num_games):
        result = simulate(whiteAgent, blackAgent)
        if result == '1-0':
            whiteWins += 1
        elif result == '0-1':
            blackWins += 1
        else:
            draws += 1

    print whiteWins, blackWins, draws
