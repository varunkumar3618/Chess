import chess
import numpy as np

PIECE_TYPES = [chess.ROOK, chess.KNIGHT, chess.BISHOP, chess.QUEEN, chess.KING, chess.PAWN]
PIECE_TYPES_WITHOUT_KING = [chess.ROOK, chess.KNIGHT, chess.BISHOP, chess.QUEEN, chess.PAWN]
DEFAULT_PIECE_SCORES = {
    chess.ROOK: 5,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.QUEEN: 9,
    chess.PAWN: 1
}

def piece_counts(board, player):
    return {
        piece: len(board.pieces(piece, player))
        for piece in PIECE_TYPES
    }

def material_value_of_player(board, player, scores=DEFAULT_PIECE_SCORES):
    value = 0
    counts = piece_counts(board, player)
    for piece in PIECE_TYPES:
        value += counts[piece] * scores[piece]
    return value

def material_value(board):
    other = chess.BLACK if board.turn is chess.WHITE else chess.WHITE
    return material_value_of_player(board, board.turn) - material_value_of_player(board, other)

class Feature(object):
    @property
    def shape(self):
        pass
    def value(self, state):
        pass

class Counts(object):
    @property
    def shape(self):
        return (10,)
    def value(self, state):
        board = state.getBoard()
        counts = []
        whiteCounts = piece_counts(board, chess.WHITE)
        for piece in PIECE_TYPES_WITHOUT_KING:
            counts.append(whiteCounts[piece])
        blackCounts = piece_counts(board, chess.BLACK)
        for piece in PIECE_TYPES_WITHOUT_KING:
            counts.append(blackCounts[piece])
        return np.array(counts, dtype='float32')