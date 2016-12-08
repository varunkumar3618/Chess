# -*- coding: utf-8 -*-

def renderBoard(board):
    pieces = {'R':'♜', 'N':'♞', 'B':'♝', 'Q':'♛', 'K':'♚', 'P':'♟',
              'r':'♖', 'n':'♘', 'b':'♗', 'q':'♕', 'k':'♔', 'p':'♙', '.':'·'}
    fen = board.fen()
    position = fen.split(" ")[0]
    rows = position.split("/")
    for row in rows:
        rowString = ""
        for pieceName in row:
            if pieceName.isdigit():
                rowString += "".join([". "] * int(pieceName))
            else:
                rowString += "{} ".format(pieces[pieceName])
        print rowString
