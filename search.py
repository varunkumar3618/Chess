import numpy as np

class SearchAlgorithm(object):
    def configure(self, valueFunction):
        pass

    def principalVariationPath(self, state, depth):
        raise NotImplementedError("Should return the principal variation path.")

class MinimaxSearch(SearchAlgorithm):
    def __init__(self):
        self.valueFunction = None
        self.cache = None
        self.configured = False

    def configure(self, valueFunction):
        if self.configured:
            raise ValueError('Already configured.')
        self.valueFunction = valueFunction
        self.cache = {}
        self.configured = True

    def principalVariationPath(self, state, depth):
        if not self.configured:
            raise ValueError('The search algorithm must be configured with a value function.')
        return list(reversed(self._principalVariationPath(state, depth, True)[1]))

    def _principalVariationPath(self, state, depth, maximize):
        key = (state, depth, maximize)
        if key not in self.cache:
            if state.isEnd():
                result = (state.getScore(), [])
            elif depth == 0:
                result = (self.valueFunction(state), [])
            elif maximize:
                valuesAndSuccessors = []
                for action in state.getActions():
                    successor = state.generateSuccessor(action)
                    value, trace = self._principalVariationPath(successor, depth, False)
                    valuesAndSuccessors.append((value, trace + [(successor, action)]))
                result = max(valuesAndSuccessors)
            else:
                valuesAndSuccessors = []
                for action in state.getActions():
                    successor = state.generateSuccessor(action)
                    value, trace = self._principalVariationPath(successor, depth - 1, True)
                    valuesAndSuccessors.append((value, trace + [(successor, action)]))
                result = min(valuesAndSuccessors)
            self.cache[key] = result
        return self.cache[key]

class MinimaxSearchWithAlphaBeta(SearchAlgorithm):
    def __init__(self):
        self.valueFunction = None
        self.cache = None
        self.configured = False

    def configure(self, valueFunction):
        if self.configured:
            raise ValueError('Already configured.')
        self.valueFunction = valueFunction
        self.cache = {}
        self.configured = True

    def principalVariationPath(self, state, depth):
        if not self.configured:
            raise ValueError('The search algorithm must be configured with a value function.')
        return list(reversed(self._principalVariationPath(state, depth, True, -float('inf'), float('inf'))[1]))

    def _principalVariationPath(self, state, depth, maximize, alpha, beta):
        key = (state, depth, maximize, alpha, beta)
        if key not in self.cache:
            if state.isEnd():
                result = (state.getScore(), [])
            elif depth == 0:
                result = (self.valueFunction(state), [])
            elif maximize:
                valuesAndSuccessors = []
                for action in state.getActions():
                    successor = state.generateSuccessor(action)
                    value, trace = self._principalVariationPath(successor, depth, False, alpha, beta)
                    valuesAndSuccessors.append((value, trace + [(successor, action)]))
                    alpha = max(alpha, value)
                    if beta <= alpha:
                        return max(valuesAndSuccessors)
                result = max(valuesAndSuccessors)
            else:
                valuesAndSuccessors = []
                for action in state.getActions():
                    successor = state.generateSuccessor(action)
                    value, trace = self._principalVariationPath(successor, depth - 1, True, alpha, beta)
                    valuesAndSuccessors.append((value, trace + [(successor, action)]))
                    beta = min(beta, value)
                    if beta <= alpha:
                        return min(valuesAndSuccessors)
                result = min(valuesAndSuccessors)
            self.cache[key] = result
        return self.cache[key]

def tictactoeTest():
    class State(object):
        cache = {}
        def __new__(cls, board, turn):
            board_tup = tuple(board.flatten())
            key = (board_tup, turn)
            if key not in State.cache:
                obj = super(State, cls).__new__(cls, board, turn)
                State.cache[key] = obj
            return State.cache[key]
        def __init__(self, board, turn):
            assert board.shape == (3, 3)
            assert turn == 'x' or 'o'
            self.board = board
            self.turn = turn
        def isEnd(self):
            return self.crossWin() or self.circleWin() or self.full()
        def getActions(self):
            actions = []
            for i in range(3):
                for j in range(3):
                    if self.board[i][j] == 0:
                        actions.append((i, j))
            return actions
        def getScore(self):
            assert self.isEnd()
            if self.crossWin():
                return -100
            elif self.circleWin():
                return 100
            else:
                return 0
        def generateSuccessor(self, action):
            newBoard = np.copy(self.board)
            if self.turn == 'x':
                newBoard[action] = -1
                return State(newBoard, 'o')
            else:
                newBoard[action] = 1
                return State(newBoard, 'x')
        def crossWin(self):
            return any(np.sum(self.board[i]) == -3 for i in range(3))\
                or any(np.sum(self.board[:, i]) == -3 for i in range(3))\
                or self.board[0, 0] + self.board[1, 1] + self.board[2, 2] == -3\
                or self.board[0, 2] + self.board[1, 1] + self.board[2, 0] == -3
        def circleWin(self):
            return any(np.sum(self.board[i]) == 3 for i in range(3))\
                or any(np.sum(self.board[:, i]) == 3 for i in range(3))\
                or self.board[0, 0] + self.board[1, 1] + self.board[2, 2] == 3\
                or self.board[0, 2] + self.board[1, 1] + self.board[2, 0] == 3
        def full(self):
            return np.sum(np.abs(self.board)) == 9
        def __repr__(self):
            lines = []
            for i in range(3):
                vals = tuple('x' if v == -1 else ('o' if v == 1 else '_') for v in self.board[i])
                lines.append('%s %s %s' % vals)
            return '\n'.join(lines)
        def __eq__(self, other):
            return isinstance(other, State) and np.array_equal(self.board, other.board) and self.turn == other.turn
    def valueFunction(state):
        if state.isEnd():
            return state.getScore()
        else:
            return 0

    for search in [MinimaxSearch(), MinimaxSearchWithAlphaBeta()]:
        search.configure(valueFunction)
        result = search.principalVariationPath(State(np.zeros((3, 3), dtype='int'), 'x'), 6)
        finalState = result[-1][0]
        assert finalState.full() and not finalState.crossWin() and not finalState.circleWin()

def prisonersTest():
    states_strs = ['', 'c', 'd', 'cd', 'cc', 'dc', 'dd']
    values = {
        '': 0,
        'c': 0,
        'd': 0,
        'cd': 2,
        'cc': -1,
        'dc': -2,
        'dd': 0
    }

    class State(object):
        def __init__(self, state_string):
            self.state_string = state_string
        def getActions(self):
            if len(self.state_string) < 2:
                return ['c', 'd']
            else:
                return []
        def isEnd(self):
            return len(self.state_string) == 2
        def generateSuccessor(self, action):
            return State(self.state_string + action)
        def getScore(self):
            if len(self.state_string) < 2:
                raise ValueError
            return values[self.state_string]
        def __repr__(self):
            return self.state_string
        def __eq__(self, other):
            return isinstance(other, State) and self.state_string == other.state_string
    def valueFunction(state):
        return values[state.state_string]

    for search in [MinimaxSearch(), MinimaxSearchWithAlphaBeta()]:
        search.configure(valueFunction)
        result = search.principalVariationPath(State(''), 1)
        expected = [(State('c'), 'c'), (State('cc'), 'c')]
        assert result == expected, 'Expected: %s, Found: %s' % (expected, result)

def test():
    prisonersTest()
    tictactoeTest()

if __name__ == '__main__':
    test()