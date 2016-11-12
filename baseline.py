import random
import chess, chess.uci

def baselinePolicy(board):
    legal_moves = list(board.legal_moves)
    return random.choice(legal_moves)

def playAgainstEngine(
        policy,
        board=chess.Board(),
        engineFile="./stockfish",
        playAsWhite=True,
        engineMovetime=20):
    """
    Arguments:
        policy: A function that takes board positions and returns a legal move.
        board: The starting board.
        engineFile: The path of the engine executable.
        playAsWhite: Whether the policy plays as white.
        engineMovetime: The time alloted to the engine per move.
    Returns:
        Result: 1 if the policy won, -1 if it lost, and 0 if the game ended in a stalemate.
    """
    engine = chess.uci.popen_engine(engineFile)
    engine.uci()
    policyPlays = playAsWhite

    while not board.is_game_over():
        if policyPlays:
            move = policy(board)
        else:
            engine.position(board)
            move = engine.go(movetime=engineMovetime)[0]
        board.push(move)
        policyPlays = not policyPlays

    engine.quit()

    return board.result()

if __name__ == '__main__':
    for i in range(5):
        print playAgainstEngine(baselinePolicy)
