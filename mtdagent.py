import collections
import chess, chess.uci

MATE_VALUE = 60000 + 8*2700

VALUES = {
    chess.PAWN: (
        198, 198, 198, 198, 198, 198, 198, 198,
        178, 198, 198, 198, 198, 198, 198, 178,
        178, 198, 198, 198, 198, 198, 198, 178,
        178, 198, 208, 218, 218, 208, 198, 178,
        178, 198, 218, 238, 238, 218, 198, 178,
        178, 198, 208, 218, 218, 208, 198, 178,
        178, 198, 198, 198, 198, 198, 198, 178,
        198, 198, 198, 198, 198, 198, 198, 198
    ),
    chess.BISHOP: (
        797, 824, 817, 808, 808, 817, 824, 797,
        814, 841, 834, 825, 825, 834, 841, 814,
        818, 845, 838, 829, 829, 838, 845, 818,
        824, 851, 844, 835, 835, 844, 851, 824,
        827, 854, 847, 838, 838, 847, 854, 827,
        826, 853, 846, 837, 837, 846, 853, 826,
        817, 844, 837, 828, 828, 837, 844, 817,
        792, 819, 812, 803, 803, 812, 819, 792
    ),
    chess.KNIGHT: (
        627, 762, 786, 798, 798, 786, 762, 627,
        763, 798, 822, 834, 834, 822, 798, 763,
        817, 852, 876, 888, 888, 876, 852, 817,
        797, 832, 856, 868, 868, 856, 832, 797,
        799, 834, 858, 870, 870, 858, 834, 799,
        758, 793, 817, 829, 829, 817, 793, 758,
        739, 774, 798, 810, 810, 798, 774, 739,
        683, 718, 742, 754, 754, 742, 718, 683
    ),
    chess.ROOK: (
        1258, 1263, 1268, 1272, 1272, 1268, 1263, 1258,
        1258, 1263, 1268, 1272, 1272, 1268, 1263, 1258,
        1258, 1263, 1268, 1272, 1272, 1268, 1263, 1258,
        1258, 1263, 1268, 1272, 1272, 1268, 1263, 1258,
        1258, 1263, 1268, 1272, 1272, 1268, 1263, 1258,
        1258, 1263, 1268, 1272, 1272, 1268, 1263, 1258,
        1258, 1263, 1268, 1272, 1272, 1268, 1263, 1258,
        1258, 1263, 1268, 1272, 1272, 1268, 1263, 1258,
    ),
    chess.QUEEN: (
        2529, 2529, 2529, 2529, 2529, 2529, 2529, 2529,
        2529, 2529, 2529, 2529, 2529, 2529, 2529, 2529,
        2529, 2529, 2529, 2529, 2529, 2529, 2529, 2529,
        2529, 2529, 2529, 2529, 2529, 2529, 2529, 2529,
        2529, 2529, 2529, 2529, 2529, 2529, 2529, 2529,
        2529, 2529, 2529, 2529, 2529, 2529, 2529, 2529,
        2529, 2529, 2529, 2529, 2529, 2529, 2529, 2529,
        2529, 2529, 2529, 2529, 2529, 2529, 2529, 2529
    ),
    chess.KING: (
        60098, 60132, 60073, 60025, 60025, 60073, 60132, 60098,
        60119, 60153, 60094, 60046, 60046, 60094, 60153, 60119,
        60146, 60180, 60121, 60073, 60073, 60121, 60180, 60146,
        60173, 60207, 60148, 60100, 60100, 60148, 60207, 60173,
        60196, 60230, 60171, 60123, 60123, 60171, 60230, 60196,
        60224, 60258, 60199, 60151, 60151, 60199, 60258, 60224,
        60287, 60321, 60262, 60214, 60214, 60262, 60321, 60287,
        60298, 60332, 60273, 60225, 60225, 60273, 60332, 60298,
    )
}

# Evalutes the board from the perspective of the white player
def evaluate(board):
    value = 0
    for pieceType, valueBoard in VALUES.items():
        for square in board.pieces(pieceType, chess.WHITE):
            value += VALUES[pieceType][square]
        for square in board.pieces(pieceType, chess.BLACK):
            value -= VALUES[pieceType][square]
    if board.is_game_over():
        if board.result == '1-0':
            value += MATE_VALUE
        elif board.result == '0-1':
            value -= MATE_VALUE
    return value


class BoardKey(object):
    def __init__(self, fen, zobrist):
        self.fen = fen
        self.zobrist = zobrist

    def __hash__(self):
        return self.zobrist

    def __eq__(self, other):
        return isinstance(other, BoardKey)\
            and self.zobrist == other.zobrist\
            and self.fen == other.fen


class LRUCache(object):
    def __init__(self, size):
        self.size = size
        self.elems = []
    def insert(self, elem):
        if elem in self.elems:
            self.elems.remove(elem)
        if len(self.elems) >= self.size:
            self.elems.pop(0)
        self.elems.append(elem)


class AlphaBetaAgent(object):
    def __init__(self, depth):
        self.depth = depth
        self.alphaBetaCache = collections.defaultdict(dict)
        self.killerCache = collections.defaultdict(lambda: LRUCache(2))

    def beginGame(self):
        pass

    def _getMoves(self, board, depth):
        legalMoves = set(board.generate_legal_moves())
        killerMoves = set(self.killerCache[depth].elems)
        for killer in killerMoves:
            if killer in legalMoves:
                yield killer
        for move in legalMoves:
            if move not in killerMoves:
                yield move

    def _addKillerMove(self, board, depth, move):
        self.killerCache[depth].insert(move)

    def _score(self, board):
        return evaluate(board)

    def _alphaBetaSearch(self, board, depth, alpha=-float('inf'), beta=float('inf')):
        key = BoardKey(board.fen(), board.zobrist_hash())
        if key in self.alphaBetaCache[depth]:
            lower, upper, move = self.alphaBetaCache[depth][key]
            if lower >= beta:
                return lower, None
            elif upper <= alpha:
                return upper, None
            alpha, beta = max(lower, alpha), min(upper, beta)
        else:
            lower, upper = -float('inf'), float('inf')

        if depth == 0:
            bestVal, bestMove = self._score(board), None
        elif board.turn == chess.WHITE:
            a, aMove = alpha, None
            bestVal, bestMove = -float('inf'), None
            for move in self._getMoves(board, depth):
                board.push(move)
                val = self._alphaBetaSearch(board, depth - 1, a, beta)[0]
                board.pop()
                if val <= a and aMove is not None:
                    self._addKillerMove(board, depth, aMove)
                if val > bestVal:
                    bestVal, bestMove = val, move
                    if bestVal > a:
                        a, aMove = bestVal, bestMove
                    if bestVal >= beta:
                        break
        else:
            b, bMove = beta, None
            bestVal, bestMove = float('inf'), None
            for move in self._getMoves(board, depth):
                board.push(move)
                val = self._alphaBetaSearch(board, depth - 1, alpha, b)[0]
                board.pop()

                if val >= b and bMove is not None:
                    self._addKillerMove(board, depth, bMove)
                if val < bestVal:
                    bestVal, bestMove = val, move
                    if bestVal < b:
                        b, bMove = bestVal, bMove
                    if bestVal <= alpha:
                        break

        if bestVal <= alpha:
            upper = bestVal
        if bestVal > alpha and bestVal < beta:
            lower = upper = bestVal
        if bestVal >= beta:
            lower = bestVal

        self.alphaBetaCache[depth][key] = (lower, upper, bestMove)
        return bestVal, bestMove

    def _mtdfSearch(self, board, guess, depth):
        lower, upper = -float('inf'), float('inf')
        move = None

        while lower < upper:
            if guess == lower:
                beta = guess + 1
            else:
                beta = guess
            guess, move = self._alphaBetaSearch(board, depth, beta - 1, beta)
            if guess < beta:
                upper = guess
            else:
                lower = guess
        return guess, move

    def _deepeningMTDFSearch(self, board, maxDepth=6):
        guess, move = 0, None
        for depth in range(1, maxDepth + 1):
            nextGuess, nextMove = self._mtdfSearch(board, guess, depth)
            if nextMove is not None:
                guess, move = nextGuess, nextMove
            else:
                break
        print guess, move
        return guess, move

    def getMove(self, board):
        return self._deepeningMTDFSearch(board, self.depth)[1]


class UCIChessAgent(object):
    def __init__(self, engineFile, engineMoveTime):
        self.engine = chess.uci.popen_engine(engineFile)
        self.engine.uci()
        self.engineMoveTime = engineMoveTime
    def beginGame(self):
        self.engine.ucinewgame()
    def getMove(self, board):
        self.engine.position(board)
        return self.engine.go(movetime=self.engineMoveTime)[0]


def simulate(whiteAgent, blackAgent, verbose=True):
    board = chess.Board(fen="8/1pp4p/8/8/1Pb2nP1/p4Pk1/7r/2R4K w - - 4 33")
    whiteAgent.beginGame()
    blackAgent.beginGame()
    while not board.is_game_over():
        if verbose:
            print board
            print board.fen()
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
        print board
        print board.reset()
        print '================================'
    return board.result()

import os
import signal
import sys
import time

def launchPDB(sig, frame):
    import pdb
    pdb.Pdb().set_trace(frame)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, launchPDB)
    print(os.getpid())
    simulate(AlphaBetaAgent(depth=4), UCIChessAgent('./engines/stockfish', 1))
