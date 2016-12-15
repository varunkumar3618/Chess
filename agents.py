import random
import collections

import chess

class Agent(object):
    def get_move(self, board):
        raise NotImplementedError("Return a move.")
    def begin_game(self):
        pass

class HumanAgent(Agent):
    pass

class RandomAgent(Agent):
    def __init__(self, name="random"):
        self.name = name
    def get_move(self, board):
        moves = list(board.generate_legal_moves())
        return random.choice(moves)

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


class MTDAgent(object):
    """
    An gent with MTD(f) search.
    """
    def __init__(self, model, depth=4, name="MTD"):
        self.name = name
        self.depth = depth
        self.alphaBetaCache = collections.defaultdict(dict)
        self.killerCache = collections.defaultdict(lambda: LRUCache(2))
        self.model = model

    def begin_game(self):
        self.alphaBetaCache.clear()
        self.killerCache.clear()

    def _get_moves(self, board, depth):
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

    def _alphaBetaSearch(self, board, depth, alpha=-float('inf'), beta=float('inf')):
        key = BoardKey(board.fen(), board.zobrist_hash())
        if key in self.alphaBetaCache[depth]:
            lower, upper, move = self.alphaBetaCache[depth][key]
            if lower > beta:
                return lower, None
            elif upper < alpha:
                return upper, None
            alpha, beta = max(lower, alpha), min(upper, beta)
        else:
            lower, upper = -float('inf'), float('inf')

        if depth == 0:
            bestVal, bestMove = self.model.evaluate(board), None
        elif board.turn == chess.WHITE:
            a, aMove = alpha, None
            bestVal, bestMove = -float('inf'), None
            for move in self._get_moves(board, depth):
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
            for move in self._get_moves(board, depth):
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
        searchPoint = guess
        step = 100
        result = None

        while lower < upper:
            # print "------------------"
            # print lower, "(search at {})".format(searchPoint), upper
            result, move = self._alphaBetaSearch(board, depth, searchPoint, searchPoint)
            # print "result: {} {}".format(result, move)
            if result > searchPoint:
                # print "Guess was too low"
                lower = result
                searchPoint = lower + step
            else:
                # print "Guess was too high"
                upper = result
                searchPoint = upper - step
            step = (upper - lower) / float(2)
            # print "Step", step
        # Special case: 0-length interval (board's upper and lower bounds are equal)
        if move is None:
            result, move = self._alphaBetaSearch(board, depth, result, result)
        # print "Final interval", lower, upper
        # print "return", result, move
        return result, move

    def _deepeningMTDFSearch(self, board, maxDepth=6):
        guess, move = 0, None
        for depth in range(1, maxDepth + 1):
            nextGuess, nextMove = self._mtdfSearch(board, guess, depth)
            if nextMove is not None:
                guess, move = nextGuess, nextMove
            else:
                break
        return guess, move

    def get_move(self, board):
        return self._deepeningMTDFSearch(board, self.depth)[1]
