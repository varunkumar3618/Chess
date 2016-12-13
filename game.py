# -*- coding: utf-8 -*-

import chess

def renderBoard(board):
    pieces = {'R':'♜', 'N':'♞', 'B':'♝', 'Q':'♛', 'K':'♚', 'P':'♟',
              'r':'♖', 'n':'♘', 'b':'♗', 'q':'♕', 'k':'♔', 'p':'♙', '.':'·'}
    fen = board.fen()
    position = fen.split(" ")[0]
    rows = position.split("/")
    final = []
    for row in rows:
        rowString = ""
        for pieceName in row:
            if pieceName.isdigit():
                rowString += "".join([". "] * int(pieceName))
            else:
                rowString += "{} ".format(pieces[pieceName])
        final.append(rowString)
    return "\n".join(final)

class Simulator(object):
    def __init__(self, verbose=False):
        self.verbose = verbose

    def display(self, *toPrint):
        if self.verbose:
            print " ".join(map(str, toPrint))

    def simulate(self, playerAgent, opponentAgent, numGames=10):
        results = []
        for game in xrange(numGames):
            playerAgent.begin_game()
            opponentAgent.begin_game()
            if game % 2 == 0:
                whiteAgent = playerAgent
                blackAgent = opponentAgent
            else:
                whiteAgent = opponentAgent
                blackAgent = playerAgent

            self.display("Game {} of {}".format(game + 1, numGames))
            self.display("White: {}, Black: {}".format(whiteAgent.name, blackAgent.name))

            board = chess.Board()
            nextBoard = chess.Board()

            self.display(renderBoard(board), "\n")
            moveNumber = 1

            while not board.is_game_over():
                if board.turn == chess.WHITE:
                    move = whiteAgent.get_move(board)
                else:
                    move = blackAgent.get_move(board)

                nextBoard.push(move)
                self.display("Move ", moveNumber)
                self.display(renderBoard(nextBoard), "\n")

                if nextBoard.is_game_over():
                    self.display("Game over", nextBoard.result())
                    self.display("-----------------------------------------------")

                    results.append(nextBoard.result())

                board.push(move)
                moveNumber += 1
        return results

def play(whiteAgent, blackAgent, verbose=10):
    simulator = Simulator(verbose=verbose)
    results = simulator.simulate(whiteAgent, blackAgent)
    return results[0]
