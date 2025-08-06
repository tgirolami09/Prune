import subprocess
import sys
from chess import Board, BLACK, WHITE, engine
from tqdm import tqdm, trange
from multiprocessing import Pool
import time
import numpy as np
movetime = int(sys.argv[3])/1000
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

def get_confidence(results):
    scores = np.arange(5)/4
    score = (results*scores).sum()
    tot = results.sum()
    percentage = score/tot
    eloDelta = eloDiff(percentage)
    resP = results/tot
    resDev = resP*(scores-percentage)**2
    stdDeviation = sqrt(resDev.sum()) / sqrt(tot)
    low = 0
    high = 1
    for i in range(5):
        mid = (low+high)/2
        x = get_delta(percentage, mid, stdDeviation)
        if x > abs(eloDelta):
            high = mid
        else:
            low = mid
    return (low+high)/2, eloDelta, get_delta(percentage, 0.95, stdDeviation)


def playGame(startFen, prog1, prog2):
    curProg, otherProg = prog1, prog2
    board = Board(startFen)
    moves = []
    while not board.is_game_over():
        result = curProg.play(board, engine.Limit(time=movetime))
        board.push(result.move)
        moves.append(result.move.uci())
        curProg, otherProg = otherProg, curProg
    if board.outcome().winner == WHITE:
        return moves, 0
    elif board.outcome().winner == BLACK:
        return moves, 1
    else:
        return moves, 2

def playBatch(args):
    id, rangeGame = args
    file = f"games{id}.log"
    with open(file, 'r') as f:
        previousGames = f.readlines()
    previousGames = previousGames[:len(previousGames)-len(previousGames)%2]
    rangeGame = range(rangeGame.start+len(previousGames), rangeGame.stop)
    log = open(file, "w")
    log.writelines(previousGames)
    log.flush()
    results = [0]*5 # (lose/lose) (lose/draw) ((lose/win) | (draw/draw)) (win/draw) (win/win)
    prog1 = engine.SimpleEngine.popen_uci(sys.argv[1])
    prog2 = engine.SimpleEngine.popen_uci(sys.argv[2])
    for idBeginBoard in rangeGame:
        beginBoard = beginBoards[idBeginBoard]
        beginBoard = beginBoard.replace('\n', '')
        interResults = [0, 0, 0]
        for idProg, prog, _prog in ((0, prog1, prog2), (1, prog2, prog1)):
            moves, winner = playGame(beginBoard, prog, _prog)
            log.write(beginBoard+' moves '+' '.join(moves)+'\n')
            log.flush()
            interResults[min(winner ^ idProg, 2)] += 1
            #print(board.outcome().winner)
            prog1.configure({'Clear Hash':None})
            prog2.configure({'Clear Hash':None})
        results[interResults[0]*2+interResults[1]] += 1
        sys.stdout.write('\n'*(id//10)+'\r'+'\t'*(id%10)*2+'/'.join(map(str, results))+'\033[F'*(id//10)+'\r')
        #sys.stdout.write('\r'+'\t'*id*2+str(round(get_confidence(results[0], results[2], results[1])[0], 5)))
        sys.stdout.flush()
    log.close()
    prog1.quit()
    prog2.quit()
    time.sleep(1)
    return np.array(results)

with open("beginBoards.out") as games:
    beginBoards = list(games.readlines())

nbProcess = 70
if not (len(sys.argv) > 4 and sys.argv[4] == "continue"):
    for i in range(nbProcess):
        with open(f'games{i}.log', "w") as f:f.write('')
nbBoards = len(beginBoards)
pool = Pool(nbProcess)
results = np.array(pool.map(playBatch, [(id, range(id*nbBoards//nbProcess, (id+1)*nbBoards//nbProcess)) for id in range(nbProcess)]))
print("\n"*((nbProcess+9)//10))
Aresults = results.sum(axis=0)
print('/'.join(map(str, Aresults)))
#thank to https://3dkingdoms.com/chess/elo.htm


confidence, eloDelta, difference = get_confidence(Aresults)
print(f"{eloDelta} +/- {difference}")
print(f"the first version is better than the second with a probability of {confidence}")
