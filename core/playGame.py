import sys
import subprocess
from chess import pgn
import chess
import time
def pushCommand(prog, command):
    prog.stdin.write(command.encode())
    prog.stdin.flush()

def readResult(prog):
    dataMoves = {}
    markEnd = 'bestmove '
    lastMate = 300
    while 1:
        line = prog.stdout.readline().decode('utf-8')
        line = line.replace('\n', '')
        if line.startswith(markEnd):
            break
        elif "currmove" not in line:
            if "mate" in line:
                n=line.split("mate")[1]
                if n >= lastMate:continue
            print(line)
    return line[len(markEnd):].split()[0]

prog1 = subprocess.Popen([sys.argv[1]], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
prog2 = subprocess.Popen([sys.argv[2]], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
movetime = int(sys.argv[3])
game = pgn.Game()
node = game
if len(sys.argv) > 4:
    board = chess.Board(sys.argv[4] if sys.argv[4] != 'startpos' else 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
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
moves = []
while not board.is_game_over() and not board.is_seventyfive_moves():
    print(board.fen())
    pushCommand(prog1, f"position fen {startFen} moves {" ".join(moves)}\n")
    pushCommand(prog1, f"go movetime {movetime}\n")
    move = readResult(prog1)
    rmove = chess.Move.from_uci(move)
    board.push(rmove)
    node = node.add_variation(rmove)
    print(board)
    prog1, prog2 = prog2, prog1
    moves.append(move)
print(board.outcome())
print(game)