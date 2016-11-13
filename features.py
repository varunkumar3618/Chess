import chess

PIECE_TYPES = [chess.ROOK, chess.KNIGHT, chess.BISHOP, chess.QUEEN, chess.KING, chess.PAWN]
DEFAULT_PIECE_SCORES = {
    chess.ROOK: 5,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.QUEEN: 9,
    chess.KING: 1000,
    chess.PAWN: 1
}

class Features(object):
    def __init__(self, board=None):
        if board is None:
            board = chess.Board()
        self.board = board

    def push(self, move):
        self.board.push(move)

    def pop(self):
        self.board.pop()

    def piece_counts(self, player):
        return {
            piece: len(self.board.pieces(piece, player))
            for piece in PIECE_TYPES
        }

    def material_value_of_player(self, player, scores=DEFAULT_PIECE_SCORES):
        value = 0
        counts = self.piece_counts(player)
        for piece in PIECE_TYPES:
            value += counts[piece] * scores[piece]
        return value

    def material_value(self):
        other = chess.BLACK if self.board.turn is chess.WHITE else chess.WHITE
        return self.material_value_of_player(self.board.turn) - self.material_value_of_player(other)

