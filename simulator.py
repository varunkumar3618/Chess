def simulate(startState, whiteRL, blackRL, numTrials=10, maxMoveCount=1000, verbose=False):
    totalRewards = []  # The rewards we get on each trial
    for trial in range(numTrials):
        whiteRL.beginSession()
        blackRL.beginSession()
        state = startState
        sequence = [state]
        totalReward = 0
        for _ in xrange(maxMoveCount):
            if state.getAgent() == 0:
                action = whiteRL.getAction(state)
            else:
                action = blackRL.getAction(state)

            successor = state.generateSuccessor(action)
            whiteReward = successor.getScore()

            sequence.append(action)
            sequence.append(whiteReward)
            sequence.append(str(successor.getBoard()))

            whiteRL.incorporateFeedback(state, action, whiteReward, successor)
            blackRL.incorporateFeedback(state, action, -whiteReward, successor)

            totalReward += whiteReward
            if successor.isEnd():
                break
            state = successor
        if verbose:
            print "Trial %d (totalReward for white = %s)" % (trial, totalReward)
            for i in xrange(0, len(sequence) - 3, 3):
                print "Board: \n", sequence[i]
                print "Action: ", sequence[i + 1]
            print "Final Board: \n", sequence[-1]
            print "Final reward: ", totalReward
        totalRewards.append(totalReward)
    return totalRewards
