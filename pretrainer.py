import os
import argparse

import numpy as np
import chess
import chess.pgn

from nn import *

class BoardToNumpyVisitor(chess.pgn.BaseVisitor):
    def __init__(self, *args):
        self._boards = []

    def begin_game(self):
        self._boards = []

    def _make_board(self, board):
        board_arr = np.zeros((2, 8, 8, 6), dtype="float32")
        for rank in range(8):
            for file in range(8):
                piece = board.piece_at(8 * rank + file)
                piece_color, piece_type = piece.color, piece.piece_type
                board_arr[piece_color][rank][file][piece_type - 1] = 1.

        # For each player, contains:
        # -> Turn: True if it's the players turn
        # -> Kingside castling rights
        # -> Queenside castling rights
        prop_arr = np.zeros((2, 3))
        for color in [chess.WHITE, chess.BLACK]:
            prop_arr[color][0] = 1. if board.turn == color else 0.
            prop_arr[color][1] = 1. if board.has_kingside_castling_rights(color) else 0.
            prop_arr[color][2] = 1. if board.has_queenside_castling_rights(color) else 0.

        return (board_arr, prop_arr)

    def visit_move(self, board, move):
        board.push(move)
        self._boards.append(self._make_board(move))
        board.pop()

    def boards(self):
        return self._boards

def make_model():
    model = Sequential(layers=[
        Linear(0.1 * np.ones((2 * , 1), dtype="float32")),
        Sigmoid()
    ])
    return model

def train_model(pgn_filenames, batch_size=2):
    model = make_model()

if __name__ == '__main__':
    pgn_filenames = []
    for filename in os.listdir("data"):
        if ".pgn" in filename:
            pgn_filenames.append(os.path.join("data", filename))
    train_model(pgn_filenames)
