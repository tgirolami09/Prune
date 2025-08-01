import subprocess
import sys
from chess import *
from tqdm import tqdm, trange
from multiprocessing import Pool
import time
movetime = int(sys.argv[3])
goCommand = f"go movetime {movetime}\n"
def pushCommand(prog, command):
    prog.stdin.write(command.encode())
    prog.stdin.flush()

def readResult(prog):
    dataMoves = {}
    markEnd = 'bestmove '
    lastMate = 300
    logs = ""
    timeLastLine = time.time()
    while 1:
        line = prog.stdout.readline().decode('utf-8')
        if line:
            timeLastLine = time.time()
        else:
            if (time.time()-timeLastLine) > 200:
                return 'h1h1', logs
            continue
        logs += line
        line = line.replace('\n', '')
        if line.startswith(markEnd):
            break
        elif "currmove" not in line:
            if "mate" in line:
                n=int(line.split("mate ")[1].split()[0])
                if n >= lastMate:continue
                lastMate = n
    return line[len(markEnd):].split()[0], logs
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
                pushCommand(prog, goCommand)
                end = time.time()
                move, logs = readResult(prog)
                if(move == "h1h1"):
                    print(f"position fen {beginBoard} moves {" ".join(moves)}\n")
                    print(logs)
                    print(prog.args)
                    print(goCommand)
                moves.append(move)
                log.write(' '+move)
                board.push(Move.from_uci(move))
                _prog, prog = prog, _prog
            log.write('\n')
            log.flush()
            winner = board.outcome().winner
            if winner is None:
                results[2] += 1
            else:
                results[(not winner) ^ idProg] += 1
            #print(board.outcome().winner)
            pushCommand(prog1, "setoption name Hash Clear\n")
            pushCommand(prog2, "setoption name Hash Clear\n")
        sys.stdout.write('\r'+'\t'*id*2+'/'.join(map(str, results)))
        sys.stdout.flush()
    log.close()
    pushCommand(prog1, 'quit\n')
    pushCommand(prog2, 'quit\n')
    time.sleep(1)
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
print(f"\nwins = {wins}, draws = {draws}, loses = {loses}")
#thank to https://3dkingdoms.com/chess/elo.htm
from math import log, sqrt, pi
def eloDiff(percentage):
    return -400 * log(1 / percentage - 1, 10)

def inverseErrorFunction(x):
    a = 8 * (pi - 3) / (3 * pi * (4 - pi))
    y = log(1 - x * x)
    z = 2 / (pi * a) + y / 2
    ret = sqrt(sqrt(z * z - y / a) - z)
    if (x < 0):
        return -ret
    return ret

def phiInv(p):
    return sqrt(2)*inverseErrorFunction(2*p-1)

score = wins+draws/2
tot = (wins+draws+loses)
percentage = score/tot
eloDelta = eloDiff(percentage)
winP = wins/tot
drawP = draws/tot
loseP = loses/tot
winsDev = winP*(1-percentage)**2
drawsDev = drawP*(0.5-percentage)**2
lossesDev = winP*(0-percentage)**2
dev = sqrt(winsDev + drawsDev + lossesDev) / sqrt(tot)
confidenceP = 0.95
minConfidenceP = (1 - confidenceP) / 2
maxConfidenceP = 1 - minConfidenceP
stdDeviation = sqrt(winsDev + drawsDev + lossesDev) / sqrt(tot)
devMin = percentage + phiInv(minConfidenceP) * stdDeviation
devMax = percentage + phiInv(maxConfidenceP) * stdDeviation
difference = eloDiff(devMax) - eloDiff(devMin)

print(f"{eloDelta} +/- {difference/2}")