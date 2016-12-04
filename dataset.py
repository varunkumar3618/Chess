import argparse
import csv

import chess
import chess.pgn


class GenVisitor(chess.pgn.BaseVisitor):
    def __init__(self, *args):
        self._boards = None
        self._lines = None
        self._result = None
        self._last_move = None

        super(GenVisitor, self).__init__(*args)

    def begin_game(self):
        self._boards = []

    def end_game(self):
        lines = []

        # Try to get the checkmate position
        board = chess.Board()
        board.set_epd(self._boards[-1])
        board.push(self._last_move)
        self._boards.append(board.epd())

        for i, board in enumerate(reversed(self._boards)):
            lines.append((board, self._result, i))
        self._lines = lines

    def visit_move(self, board, move):
        self._boards.append(board.epd())
        self._last_move = move

    def visit_result(self, result):
        self._result = result

    def result(self):
        return self._result

    def getLines(self):
        return self._lines


def generate_data(filename, keep_draws=False):
    lines = []
    visitor = GenVisitor()

    f = open(filename)

    offsets = []
    for offset, _ in chess.pgn.scan_headers(f):
        offsets.append(offset)

    for offset in offsets:
        f.seek(offset)
        game = chess.pgn.read_game(f)
        game.accept(visitor)

        if not keep_draws and visitor.result() == "0-0":
            continue
        lines += visitor.getLines()

    f.close()
    return lines


def main(args):
    infiles = args.infiles
    outfile = args.outfile
    keep_draws = args.keep_draws

    lines = []
    for file in infiles:
        lines += generate_data(file, keep_draws=keep_draws)

    with open(outfile, "wb") as outf:
        writer = csv.writer(outf, delimiter=",", quotechar="\"")
        for row in lines:
            writer.writerow(row)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a dataset of board positions and results from games in PGN format.")
    parser.add_argument("--infiles", type=str, nargs="+", help="PGN files to process", required=True)
    parser.add_argument("--outfile", type=str, help="The file in which to store the dataset", required=True)
    parser.add_argument("--keep_draws", type=bool, default=False, help="Whether ")

    main(parser.parse_args())


    args = parser.parse_args()