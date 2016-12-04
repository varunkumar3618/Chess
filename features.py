import chess
import numpy as np

#PIECE_TYPES = [chess.ROOK, chess.KNIGHT, chess.BISHOP, chess.QUEEN, chess.KING, chess.PAWN]
PIECE_TYPES = [chess.KING, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]
# PIECE_COUNTS for PIECE_TYPES, in order
PIECE_COUNTS = [1, 1, 2, 2, 2, 8]
PIECE_TYPES_WITHOUT_KING = [chess.ROOK, chess.KNIGHT, chess.BISHOP, chess.QUEEN, chess.PAWN]
PIECE_COLORS = [chess.WHITE, chess.BLACK]
DEFAULT_PIECE_SCORES = {
    chess.ROOK: 5,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.QUEEN: 9,
    chess.PAWN: 1,
    # need to give king *some* value for PieceLists' LVA
    chess.KING: 100
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
    def value(self, board):
        pass
    def initial_weight(self):
        raise NotImplementedError('Numpy array.')

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

class PawnOccupation(object):
    @property
    def shape(self):
        return (8, 8)
    def value(self, state):
        value = np.zeros(64, dtype='float32')
        board = state.getBoard()
        for square in board.pieces(chess.PAWN, chess.WHITE):
            value[square] = 1.
        for square in board.pieces(chess.PAWN, chess.BLACK):
            value[square] = -1.
        return value.reshape((8, 8))


##### Begin implementation of Giraffe feature extractor #####
#############################################################

class SideToMove(Feature):
    @property
    def shape(self):
        return (1, 1)
    def value(self, board):
        value = np.zeros(1, dtype='float32')
        if board.turn:
            value[0] = 1.0
        return value

class CastlingRights(Feature):
    @property
    def shape(self):
        return (2, 2)
    def value(self, board):
        value = np.zeros(4, dtype='float32');
        if (board.has_kingside_castling_rights(chess.WHITE)): value[0] = 1.0
        if (board.has_queenside_castling_rights(chess.WHITE)): value[1] = 1.0
        if (board.has_kingside_castling_rights(chess.BLACK)): value[2] = 1.0
        if (board.has_queenside_castling_rights(chess.BLACK)): value[3] = 1.0
        return value.reshape((2, 2))


# returns a 2x6 matrix, indicating counts for:
#         King  |  Queen  |  Rook  | Bishop   |  Knight  |  Pawn
# White:        |         |        |          |          |
# Black:        |         |        |          |          |
class MaterialConfiguration(Feature):
    @property
    def shape(self):
        return (2, 6)
    def value(self, board):
        value = np.zeros((2, 6), dtype='float32')
        for row, player in enumerate(PIECE_COLORS):
            counts = piece_counts(board, player)
            for col, piece in enumerate(PIECE_TYPES):
                value[row][col] = counts[piece]

        return value

# helper fn that returns a list of pieces

# Each of the 32 possible pieces has a "slot." These form the rows of the piece
# list table.  Each row is of the form:
# | piece present? | normalized x-coordinate (row) | normalized y-coordinate(col) | LVA |
#
# where LVA is "lowest-valued attacker," the *value* of the lowest-valued attacker of the
# given slot
class PieceLists(Feature):
    @property
    def shape(self):
        return (16 * 2, 4)
    def value(self, board):
        value = np.zeros((16 * 2, 4), dtype='float32')

        table_row = -1 # start @ -1 b/c the index is incremented before its first use
        for i, color in enumerate(PIECE_COLORS):
            for j, piece_type in enumerate(PIECE_TYPES):
                square_set = board.pieces(piece_type, color)

                square = None
                for k in range(PIECE_COUNTS[j]):
                    try:
                        square = square_set.pop()
                    except KeyError:
                        continue
                    finally:
                        table_row += 1

                    # do something with square, piece
                    piece = board.piece_at(square)

                    # now, update the table slot (row)
                    value[table_row][0] = 1.0  # this piece must exist
                    value[table_row][1] = chess.file_index(square) / 7.0   # normalized x-coord
                    value[table_row][2] = chess.rank_index(square) / 7.0   # normalized y-coord

                    # compute the *normalized* value of the lowest-valued attacker
                    attackers_ss = board.attackers(not color, square)
                    lva = float('inf')
                    while True:
                        try:
                            attacker_square = attackers_ss.pop()
                            attacker_piece = board.piece_at(attacker_square)

                            lva = min(lva, DEFAULT_PIECE_SCORES[attacker_piece.piece_type] / 100.0)
                        except KeyError:
                            break

                    if lva < float('inf'):
                        value[table_row][3] = lva

        return value


# return an array of every feature extractor
ALL_FEATURES = [SideToMove(), CastlingRights(), MaterialConfiguration(), PieceLists()]
