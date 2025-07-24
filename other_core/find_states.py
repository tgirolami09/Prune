import sys
import subprocess
from chess import *
depth = int(sys.argv[3])
prog1 = subprocess.Popen([sys.argv[1]], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
prog2 = subprocess.Popen([sys.argv[2]], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
file = sys.argv[4]
if len(sys.argv) > 5:
    startFen = sys.argv[5]
else:
    startFen = None
try:
    def pushCommand(prog, command):
        prog.stdin.write(command.encode())
        prog.stdin.flush()

    def readResult(prog):
        dataMoves = {}
        while 1:
            line = prog.stdout.readline().decode('utf-8')
            #print(line)
            line = line.replace('\n', '')
            if line.startswith('Nodes searched: '):
                break
            else:
                L = line.split(": ")
                if len(L) == 2 and (len(L[0]) == 4 or len(L[0]) == 5) and L[1].isdigit():
                    dataMoves[L[0]] = L[1]
        return dataMoves

    S = set()
    def search(board, depth):
        fen = board.fen()
        if fen in S:
            return
        S.add(fen)
        commFen = f"position fen {board.fen()}\n"
        commPerft = f"go perft {depth}\n"
        pushCommand(prog1, commFen)
        pushCommand(prog2, commFen)

        pushCommand(prog1, commPerft)
        pushCommand(prog2, commPerft)

        moves1 = readResult(prog1)
        moves2 = readResult(prog2)
        if len(moves1) != len(moves2):
            with open(file, 'a') as f:
                f.write(fen+' : ')
                for move in (set(moves1.keys())-set(moves2.keys())):
                    f.write(' +'+move) # move not present in the second prog
                for move in (set(moves2.keys())-set(moves1.keys())):
                    f.write(' -'+move) # move not present in the first prog
                f.write('\n')
        for move, occ in moves1.items():
            if move in moves2 and moves2[move] != occ:
                board.push(Move.from_uci(move))
                search(board, depth-1)
                board.pop()
    with open(file, 'w') as f:
        f.write('') # clear file
    if startFen is not None:
        board = Board(startFen)
    else:
        board = Board()
    search(board, depth)
finally:
    pushCommand(prog1, "quit\n")
    pushCommand(prog2, "quit\n")