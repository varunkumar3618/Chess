import random

class SearchAlgorithm(object):
    def principalVariationPath(self, valueFunction, state, depth):
        raise NotImplementedError("Should return the principal variation path.")

class MinimaxSearch(SearchAlgorithm):
    def principalVariationPath(self, valueFunction, state, depth):
        return self._principalVariationPath(valueFunction, state, depth, True)[1]

    def _principalVariationPath(self, valueFunction, state, depth, maximize):
        if state.isEnd():
            return (state.getScore(), [])
        elif depth == 0:
            return (valueFunction(state), [])
        elif maximize:
            valuesAndSuccessors = []
            for action in state.getActions():
                successor = state.generateSuccessor(action)
                value, trace = self._principalVariationPath(valueFunction, successor, depth, False)
                valuesAndSuccessors.append((value, [(successor, action)] + trace))
            return max(valuesAndSuccessors)
        else:
            valuesAndSuccessors = []
            for action in state.getActions():
                successor = state.generateSuccessor(action)
                value, trace = self._principalVariationPath(valueFunction, successor, depth - 1, True)
                valuesAndSuccessors.append((value, [(successor, action)] + trace))
            return min(valuesAndSuccessors)


def test():
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
    def valueFunction(state):
        return values[state.state_string]
    print MinimaxSearch().principalVariationPath(valueFunction, State(''), 1)[0][1]

if __name__ == '__main__':
    test()