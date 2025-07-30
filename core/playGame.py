import sys
import subprocess
from chess import pgn
import chess
import time
startpos = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
def pushCommand(prog, command):
    prog.stdin.write(command.encode())
    prog.stdin.flush()

def readResult(prog):
    dataMoves = {}
    markEnd = 'bestmove '
    lastMate = 300
    line = ""
    while 1:
        lastLine = line
        line = prog.stdout.readline().decode('utf-8')
        line = line.replace('\n', '')
        if line.startswith(markEnd):
            print(line)
            break
        elif "currmove" not in line:
            if "mate" in line:
                n=int(line.split("mate ")[1].split()[0])
                if n >= lastMate:continue
                lastMate = n
            print(line)
    nodes = int(lastLine.split("nodes ")[1].split()[0])
    return line[len(markEnd):].split()[0], nodes

prog1 = subprocess.Popen([sys.argv[1]], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
prog2 = subprocess.Popen([sys.argv[2]], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
nodes1 = nodes2 = 0
movetime = int(sys.argv[3])
if len(sys.argv) > 4:
    board = chess.Board(sys.argv[4] if sys.argv[4] != 'startpos' else startpos)
    if(len(sys.argv) > 6):
        sideLimit = int(sys.argv[5])
        EloLimit = int(sys.argv[6])
        if sideLimit:prog = prog2
        else: prog = prog1
        pushCommand(prog, "setoption name UCI_limitStrength value true\n")
        pushCommand(prog, f"setoption name UCI_Elo value {EloLimit}\n")
else:
    board = chess.Board()
startFen = board.fen()
if startFen != startpos:
    game = pgn.Game(dict(Variant="From Position", FEN=startFen))
else:
    game = pgn.Game()
node = game
moves = []
while not board.is_game_over() and not board.is_seventyfive_moves():
    print(board.fen())
    pushCommand(prog1, f"position fen {startFen} moves {" ".join(moves)}\n")
    pushCommand(prog1, f"go movetime {movetime}\n")
    move, nbNodes = readResult(prog1)
    rmove = chess.Move.from_uci(move)
    board.push(rmove)
    node = node.add_variation(rmove)
    print(board)
    nodes1 += nbNodes
    prog1, prog2 = prog2, prog1
    nodes1, nodes2 = nodes2, nodes1
    moves.append(move)
print(game)
winner = board.outcome().winner
if winner == chess.WHITE:
    print(sys.argv[1])
elif winner == chess.BLACK:
    print(sys.argv[2])
else:
    print('draw')
print(prog1.args, nodes1)
print(prog2.args, nodes2)