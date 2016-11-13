import random

class Agent:
    """
    An agent must define a getAction method, but may also define the
    following methods which will be called if they exist:

    def registerInitialState(self, state): # inspects the starting state
    """
    def getAction(self, state):
        raise NotImplementedError("Override me")

class MultiAgentSearchAgent(Agent):
    def __init__(self, evalFn, depth=2):
        self.evaluationFunction = evalFn
        self.depth = depth

class MinimaxAgent(MultiAgentSearchAgent):
    def __init__(self, protagonist):
        """
        The protagonist is the agent we implement, in constrast to the reference
        chess engine. For example, if we are implementing an agent to play as
        white, the protagonist is 0 (corresponding to the white player).
        The protagonist is the agent number for which we maximize expected
        value in minimax.
        """
        self.protagonist = protagonist

    def isEnd(self, state):
        return state.getBoard().is_game_over()

    def Vmaxmin(self, state, depth):
        agent = state.getAgent()
        actions = state.getLegalActions(agent)
        # Shuffle actions to break ties randomly in max and min below
        random.shuffle(actions)
        successorsAndActions = [(state.generateSuccessor(action), action) for action in actions]
        if self.isEnd(state):
            return (state.getScore(), None)
        elif depth == 0:
            return (self.evaluationFunction(state), None)
        else:
            if agent < state.getNumAgents() - 1:
                Vdepth = depth
            else:
                Vdepth = depth - 1
            valuesAndActions = [(self.Vmaxmin(successor, Vdepth)[0], action) \
                                for successor, action in successorsAndActions]
            if agent == self.protagonist:
                # protagonist
                value, action = max(valuesAndActions)
            else:
                # opponents
                value, action = min(valuesAndActions)
            return (value, action)

    def getAction(self, gameState):
        _, action = self.Vmaxmin(gameState, self.depth)
        return action
