import sys
import os
import chess

from agent import Agent
os.path.dirname(os.path.realpath(__file__))

sys.path.insert(0,
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "engines",
        "sunfish"
    )
)
import sunfish

class SunfishAgent(Agent):
    def __init__(self, engineMoveTime):
        self.searcher = None
        self.engineMoveTime = engineMoveTime
        # Create sunfish-position-to-UCI lookup tuple
        rankGen = lambda rank: tuple(chr(ord("a") + i) + str(rank) for i in xrange(8))
        board = ()
        for i in xrange(7, -1, -1):
            board += (None,) + rankGen(i + 1) + (None,)
        board = (None,) * 20 + board + (None,) * 20
        self.sunfishToUCI = board
        # Create UCI-to-sunfish-position lookup table
        self.uciToSunfish = {}
        for index, square in enumerate(self.sunfishToUCI):
            if square is not None:
                self.uciToSunfish[square] = index

    def beginEpisode(self, state=None, params=None):
        self.searcher = sunfish.Searcher()

    def chessGameStateToSunfishPosString(self, state):
        fen = state.getBoard().board_fen()
        # e.g. rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR
        bufRank = (" " * 9 + "\n") * 2 # Buffer ranks
        sunfishPos = ""
        for rank in fen.split("/"):
            sunfishPos += " " # Buffer at left of row
            for square in rank:
                try:
                    # An int in FEN represents a number of empty spaces
                    empties = int(square)
                    sunfishPos += "." * empties
                except ValueError:
                    # Not an int, so it must be a piece
                    sunfishPos += square
            sunfishPos += "\n"
        sunfishPos = bufRank + sunfishPos + bufRank
        # Sunfish always plays as white (capitals) so if it's black's turn we
        # have to make black look like white
        if state.getBoard().turn == chess.BLACK:
            sunfishPos = sunfishPos[:].swapcase()
        return sunfishPos

    def chessGameStateToSunfishCastlingTuples(self, state):
        # Castling rights in FEN format
        xfen = state.getBoard().castling_xfen()
        whiteCastle = (
            True if "Q" in xfen else False,
            True if "K" in xfen else False
        )
        blackCastle = (
            True if "k" in xfen else False,
            True if "q" in xfen else False
        )
        return (whiteCastle, blackCastle)

    def sunfishMoveToPythonChessMove(self, sunfishMove):
        # Integer sunfish positions
        sunfishFrom = sunfishMove[0]
        sunfishTo = sunfishMove[1]
        uci = self.sunfishToUCI[sunfishFrom] + self.sunfishToUCI[sunfishTo]
        return chess.Move.from_uci(uci)

    # def pythonChessMoveToSunfishMove(self, pythonChessMove):
    #     uci = pythonChessMove.uci()
    #     if not uci:
    #         sunfishMove = None
    #     else:
    #         moveFrom = uci[:2]
    #         moveTo = uci[2:]
    #         sunfishMove = (self.uciToSunfish[moveFrom], self.uciToSunfish[moveTo])
    #     return sunfishMove

    # def chessGameStateToSunfishScore(self, state):
    #     # Copy board
    #     board = chess.Board(state.getBoard().fen())
    #     # Get all moves in reverse order
    #     moves = []
    #     while True:
    #         try:
    #             moves.append(board.pop())
    #         except IndexError:
    #             break
    #     moves = reversed(moves)
    #     print [str(move) + "\n" for move in moves]
    #     # Start with initial sunfish position
    #     pos = sunfish.Position(sunfish.initial, 0, (True,True), (True,True), 0, 0)
    #     # Replay moves and accumulate score
    #     for move in moves:
    #         sunfishMove = self.pythonChessMoveToSunfishMove(move)
    #         pos = pos.move(sunfishMove)
    #     # Now the position has been replayed from the start. It contains the score
    #     return pos.score

    def getAction(self, state):
        posString = self.chessGameStateToSunfishPosString(state)
        whiteCastle, blackCastle = self.chessGameStateToSunfishCastlingTuples(state)
        # score = self.chessGameStateToSunfishScore(state)
        # board, score, white's (queenCastle, kingCastle), black's (kingCastle, queenCastle), enPassantSquare, kingPassantSquare
        pos = sunfish.Position(posString, 0, whiteCastle, blackCastle, 0, 0)
        move, _ = self.searcher.search(pos, secs=self.engineMoveTime / 1000.0)
        newPos = pos.move(move)
        sunfish.print_pos(pos)
        sunfish.print_pos(newPos.rotate())
        return self.sunfishMoveToPythonChessMove(move)

    def incorporateFeedback(self, state, action, reward, newState):
        pass

    def endEpisode(self):
        pass
