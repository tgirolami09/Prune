import subprocess
import sys
from chess import *
from tqdm import tqdm, trange
from multiprocessing import Pool
import time
movetime = int(sys.argv[3])
goCommand = f"go movetime {movetime}\n"

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


def get_delta(percentage, confidenceP, stdDeviation):
    minConfidenceP = (1 - confidenceP) / 2
    maxConfidenceP = 1 - minConfidenceP
    devMin = percentage + phiInv(minConfidenceP) * stdDeviation
    devMax = percentage + phiInv(maxConfidenceP) * stdDeviation
    difference = eloDiff(devMax) - eloDiff(devMin)
    return difference/2

def get_confidence(wins, draws, losses):
    if(losses == 0 and draws == 0):return 1, 400, 0
    if(wins == 0 and draws == 0):return 0, -400, 0
    score = wins+draws/2
    tot = (wins+draws+losses)
    percentage = score/tot
    eloDelta = eloDiff(percentage)
    winP = wins/tot
    drawP = draws/tot
    loseP = losses/tot
    winsDev = winP*(1-percentage)**2
    drawsDev = drawP*(0.5-percentage)**2
    lossesDev = winP*(0-percentage)**2
    dev = sqrt(winsDev + drawsDev + lossesDev) / sqrt(tot)
    stdDeviation = sqrt(winsDev + drawsDev + lossesDev) / sqrt(tot)
    low = 0
    high = 1
    for i in range(5):
        mid = (low+high)/2
        x = get_delta(percentage, mid, stdDeviation)
        if x > abs(eloDelta):
            high = mid
        else:
            low = mid
    if(eloDelta < 0):
        low = 1-low
        hiwh = 1-high
    return (low+high)/2, eloDelta, get_delta(percentage, 0.95, stdDeviation)

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
            if (time.time()-timeLastLine) > movetime*2:
                return 'h1h1', logs+line
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
    return line[len(markEnd):].split()[0], logs, 'nothing'

def playGames(args):
    id, rangeGame = args
    log = open(f"games{id}.log", "w")
    results = [0]*3 # wins/loses/draw
    prog1 = progs1[id]
    prog2 = progs2[id]
    for idBeginBoard in rangeGame:
        beginBoard = beginBoards[idBeginBoard]
        beginBoard = beginBoard.replace('\n', '')
        for idProg, prog, _prog in ((0, prog1, prog2), (1, prog2, prog1)):
            gameFinished = False
            while not gameFinished:
                board = Board(beginBoard)
                moves = []
                while not board.is_game_over() and not board.is_seventyfive_moves() and not board.is_fivefold_repetition():
                    joinMoves = " ".join(moves)
                    pushCommand(prog, f"position fen {beginBoard} moves {joinMoves}\n")
                    start = time.time()
                    pushCommand(prog, goCommand)
                    end = time.time()
                    move, logs, msg = readResult(prog)
                    if(move == "h1h1"):
                        print(f"\nposition fen {beginBoard} moves {joinMoves}\n")
                        print(logs)
                        print(prog.args)
                        print(goCommand)
                        print(msg)
                        time.sleep(1)
                        pushCommand(prog1, 'quit\n')
                        pushCommand(prog2, 'quit\n')
                        prog1.wait()
                        prog2.wait()
                        return
                    moves.append(move)
                    board.push(Move.from_uci(move))
                    _prog, prog = prog, _prog
                else:
                    gameFinished = True
            log.write(beginBoard+' moves '+' '.join(moves)+'\n')
            log.flush()
            winner = board.outcome().winner
            if winner is None:
                results[2] += 1
            else:
                results[(not winner) ^ idProg] += 1
            #print(board.outcome().winner)
            pushCommand(prog1, "setoption name Hash Clear\n")
            pushCommand(prog2, "setoption name Hash Clear\n")
        sys.stdout.write('\n'*(id//10)+'\r'+'\t'*(id%10)*2+'/'.join(map(str, (results[0], results[2], results[1])))+'\033[F'*(id//10))
        #sys.stdout.write('\r'+'\t'*id*2+str(round(get_confidence(results[0], results[2], results[1])[0], 5)))
        sys.stdout.flush()
    log.close()
    pushCommand(prog1, 'quit\n')
    pushCommand(prog2, 'quit\n')
    prog1.wait()
    prog2.wait()
    time.sleep(1)
    return results

with open("beginBoards.out") as games:
    beginBoards = list(games.readlines())

nbProcess = 70
progs1 = [subprocess.Popen([sys.argv[1]], stdin=subprocess.PIPE, stdout=subprocess.PIPE) for i in range(nbProcess)]
progs2 = [subprocess.Popen([sys.argv[2]], stdin=subprocess.PIPE, stdout=subprocess.PIPE) for i in range(nbProcess)]
nbBoards = len(beginBoards)
pool = Pool(nbProcess)
results = pool.map(playGames, [(id, range(id*nbBoards//nbProcess, (id+1)*nbBoards//nbProcess)) for id in range(nbProcess)])
print("\n"*((nbProcess+9)//10))
wins = 0
loses = 0
draws = 0
for result in results:
    wins += result[0]
    loses += result[1]
    draws += result[2]
print(f"\nwins = {wins}, draws = {draws}, loses = {loses}")
#thank to https://3dkingdoms.com/chess/elo.htm


confidence, eloDelta, difference = get_confidence(wins, draws, loses)
print(f"{eloDelta} +/- {difference}")
print(f"the first version is better than the second with a probability of {confidence}")
