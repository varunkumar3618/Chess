import random

class SearchAlgorithm(object):
    def principalVariationPath(self, valueFunction, state, depth):
        raise NotImplementedError("Should return the principal variation path.")

class MinimaxSearch(SearchAlgorithm):
    def principalVariationPath(self, valueFunction, state, depth):
        return self._principalVariationPath(valueFunction, state, depth, True)

    def _principalVariationPath(self, valueFunction, state, depth, minimize):
        actions = state.getActions()

        random.shuffle(actions)
        successors = [(state.generateSuccessor(action), action) for action in actions]
        valuesAndSuccessors = [(valueFunction(successor), successor, action) for successor, action in successors]

        if state.isEnd():
            return []
        elif depth == 0:
            return []
        elif minimize:
            minValue, minSucc, minAction = min(valuesAndSuccessors)
            return [(minSucc, minAction)] + self._principalVariationPath(valueFunction, minSucc, depth, False)
        else:
            maxValue, maxSucc, maxAction = max(valuesAndSuccessors)
            return [(maxSucc, maxAction)] + self._principalVariationPath(valueFunction, maxSucc, depth - 1, True)