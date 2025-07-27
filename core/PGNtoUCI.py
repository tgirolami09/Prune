import chess
import chess.pgn as pgn
import sys
def load_game(name):
    return pgn.read_game(open(name))

game = load_game(sys.argv[1])
with open(sys.argv[2], 'w') as f:
    for move in game.mainline_moves():
        f.write(move.uci()+' ')
