import subprocess
import sys
from chess import *
from tqdm import tqdm, trange
from multiprocessing import Pool
import time
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
def playGames(args):
    id, rangeGame = args
    log = open(f"games{id}.log", "w")
    results = [0]*3 # wins/loses/draw
    prog1 = subprocess.Popen([sys.argv[1]], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    prog2 = subprocess.Popen([sys.argv[2]], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    for idBeginBoard in rangeGame:
        beginBoard = beginBoards[idBeginBoard]
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
                results[(not winner) ^ idProg] += 1
            #print(board.outcome().winner)
    print(results)
    log.close()
    pushCommand(prog1, 'quit\n')
    pushCommand(prog2, 'quit\n')
    return results

with open("beginBoards.out") as games:
    beginBoards = list(games.readlines())
nbProcess = 10
nbBoards = len(beginBoards)
pool = Pool(nbProcess)
results = pool.map(playGames, [(id, range(id*nbBoards//nbProcess, (id+1)*nbBoards//nbProcess)) for id in range(nbProcess)])
wins = 0
loses = 0
draws = 0
for result in results:
    wins += result[0]
    loses += result[1]
    draws += result[2]
print(f"wins = {wins}, draws = {draws}, loses = {loses}")