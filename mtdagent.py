import chess
PIECE_TYPES = [chess.ROOK, chess.KNIGHT, chess.BISHOP, chess.QUEEN, chess.KING, chess.PAWN]
PIECE_TYPES_WITHOUT_KING = [chess.ROOK, chess.KNIGHT, chess.BISHOP, chess.QUEEN, chess.PAWN]
PIECE_SCORES = {
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

class Position(object):
    def __init__(self, board):
        self.board = board.copy()

    def children(self):
        nextBoard = self.board.copy()
        for move in self.board.generate_legal_moves():
            nextBoard.push(move)
            yield Position(nextBoard)
            nextBoard.pop()

    def __hash__(self):
        return self.board.zobrist_hash()

class Agent(object):
    SEARCH_WINDOW = 10
    CHECKMATE_SCORE = 1e6

    def __init__(self):
        self.transTable = {}
        self.color = chess.WHITE

    def _clear(self):
        self.transTable = {}

    def beginEpisode(self, board, color):
        self.startGame(color)

    def startGame(self, color):
        if self.color != color:
            self._clear()
            self.color = color

    def _scoringFunction(self, pos):
        board = pos.board
        score = 0.
        whiteCounts = piece_counts(board, chess.WHITE)
        blackCounts = piece_counts(board, chess.BLACK)
        for piece in PIECE_TYPES_WITHOUT_KING:
            score += whiteCounts[piece] * PIECE_SCORES[piece]
            score -= blackCounts[piece] * PIECE_SCORES[piece]

        # for i in range(64):
        #     piece_type = board.piece_type_at(chess.SQUARES[i])
        #     if piece_type is chess.PAWN:
        #         color = board.piece_at(i).color
        #         if color is chess.WHITE:
        #             score += int(PAWN_BONUS[i])
        #         else:
        #             score -= int(PAWN_BONUS[i])

        if self.color is chess.WHITE:
            return score
        else:
            return -score

    def _toMaximize(self, pos):
        if pos.board.turn == self.color:
            return True
        else:
            return False

    def _alphaBetaSearch(self, pos, alpha, beta, depth):
        key = (pos, depth)
        if key in self.transTable:
            lower, upper = self.transTable[key]
            if lower >= beta:
                return lower
            elif upper <= alpha:
                return upper
            alpha = max(lower, alpha)
            beta = min(upper, beta)
        else:
            lower, upper = -float('inf'), float('inf')

        if depth == 0:
            g = self._scoringFunction(pos)
        elif self._toMaximize(pos):
            a = alpha
            g = -float('inf')
            for child in pos.children():
                if g >= beta:
                    break
                g = self._alphaBetaSearch(child, a, beta, depth - 1)
                a = max(a, g)
        else:
            b = beta
            g = float('inf')
            for child in pos.children():
                if g <= alpha:
                    break
                g = self._alphaBetaSearch(child, alpha, b, depth - 1)
                b = min(b, g)

        if g <= alpha:
            upper = g
        if g > alpha and g < beta:
            lower = upper = g
        if g >= alpha:
            lower = g

        self.transTable[key] = (lower, upper)
        return g

    def _mtdfSearch(self, pos, guess, depth):
        g = guess
        lower, upper = -Agent.CHECKMATE_SCORE, Agent.CHECKMATE_SCORE

        while lower + Agent.SEARCH_WINDOW < upper:
            if g == lower:
                beta = lower + (upper - lower) / 2
            else:
                beta = g
            g = self._alphaBetaSearch(pos, lower, beta, depth)

            if g < beta:
                upper = g
            else:
                lower = g
        return g

    def evaluate(self, pos, maxDepth=6):
        guess = 0
        for depth in range(1, maxDepth + 1):
            guess = self._mtdfSearch(pos, guess, depth)
        return guess

    def getAction(self, gamestate):
        board = gamestate.getBoard().copy()
        cands = []
        for move in board.generate_legal_moves():
            board.push(move)
            value = self.evaluate(Position(board), maxDepth=4)
            cands.append((value, move))
            board.pop()
        print cands
        return max(cands)[1]

    def incorporateFeedback(self, *args):
        pass