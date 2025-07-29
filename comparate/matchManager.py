import subprocess
import sys
from chess import *
from tqdm import tqdm, trange
import time
prog1 = subprocess.Popen([sys.argv[1]], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
prog2 = subprocess.Popen([sys.argv[2]], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
movetime = int(sys.argv[3])
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
                n=int(line.split("mate ")[1].split()[0])
                if n >= lastMate:continue
                lastMate = n
    return line[len(markEnd):].split()[0]

log = open("games.log", 'w')
with open("beginBoards.out") as games:
    results = [0]*3 # wins/loses/draw
    beginBoards = list(games.readlines())
    for beginBoard in tqdm(beginBoards):
        beginBoard = beginBoard.replace('\n', '')
        for idProg, prog, _prog in ((0, prog1, prog2), (1, prog2, prog1)):
            board = Board(beginBoard)
            moves = []
            log.write(beginBoard+' moves')
            while not board.is_game_over() and not board.is_seventyfive_moves() and not board.is_fivefold_repetition():
                pushCommand(prog, f"position fen {beginBoard} moves {" ".join(moves)}\n")
                start = time.time()
                pushCommand(prog, f"go movetime {movetime}\n")
                end = time.time()
                move = readResult(prog)
                moves.append(move)
                log.write(' '+move)
                log.flush()
                board.push(Move.from_uci(move))
                _prog, prog = prog, _prog
            log.write('\n')
            winner = board.outcome().winner
            if winner is None:
                results[2] += 1
            else:
                results[winner ^ idProg] += 1
            #print(board.outcome().winner)
    print(results)
pushCommand(prog1, 'quit\n')
pushCommand(prog2, 'quit\n')