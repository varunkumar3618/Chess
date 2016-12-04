#!/usr/local/bin/python
from features import *
import chess
import numpy as np

START_BOARD = chess.Board()
TEST1_BOARD = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
TEST2_BOARD = chess.Board("rnb1k2r/ppp2ppp/5n2/3q4/1b1P4/2N5/PP3PPP/R1BQKBNR w KQkq - 3 7")
TEST3_BOARD = chess.Board("7k/8/8/4p3/3P4/8/8/K7 w KQkq - 0 10")


pl = PieceLists()

print pl.shape
print TEST2_BOARD
print pl.value(TEST2_BOARD)

print TEST3_BOARD
print pl.value(TEST3_BOARD)
