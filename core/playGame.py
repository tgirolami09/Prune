import sys
import subprocess
from chess import pgn, engine
import chess
import time
startpos = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'

prog1 = engine.SimpleEngine.popen_uci(sys.argv[1])
prog2 = engine.SimpleEngine.popen_uci(sys.argv[2])
nodes1 = nodes2 = 0
movetime = int(sys.argv[3])
if len(sys.argv) > 4:
    board = chess.Board(sys.argv[4] if sys.argv[4] != 'startpos' else startpos)
    if(len(sys.argv) > 6):
        sideLimit = int(sys.argv[5])
        EloLimit = int(sys.argv[6])
        if sideLimit:prog = prog2
        else: prog = prog1
        prog.configure({'UCI_limitStrength':True})
        prog.configure({'UCI_Elo':EloLimit})
else:
    board = chess.Board()
startFen = board.fen()
if startFen != startpos:
    game = pgn.Game(dict(Variant="From Position", FEN=startFen))
else:
    game = pgn.Game()
node = game
moves = []
player1 = sys.argv[1]
player2 = sys.argv[2]
while not board.is_game_over() and not board.is_seventyfive_moves():
    print(board.fen())
    print(' '.join(i.uci() for i in board.move_stack))
    result = prog1.play(board, limit=engine.Limit(time=0.1), info=engine.INFO_ALL)
    rmove = result.move
    nbNodes = result.info['nodes']
    board.push(rmove)
    node = node.add_variation(rmove)
    print(board)
    nodes1 += nbNodes
    prog1, prog2 = prog2, prog1
    nodes1, nodes2 = nodes2, nodes1
    player1, player2 = player2, player1
game.headers["White"] = sys.argv[1]
game.headers["Black"] = sys.argv[2]
print(game)
winner = board.outcome().winner
if winner == chess.WHITE:
    print(sys.argv[1])
elif winner == chess.BLACK:
    print(sys.argv[2])
else:
    print('draw')
print(player1, nodes1)
print(player2, nodes2)
prog1.quit()
prog2.quit()