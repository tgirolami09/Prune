import subprocess
import sys
from chess import Board, BLACK, WHITE, engine
from tqdm import tqdm, trange
from multiprocessing import Pool
from multiprocessing.managers import SharedMemoryManager
import time
import numpy as np
isMoveTime = False
timeControl = sys.argv[3]

#seconds+seconds
if '+' in sys.argv[3]:
    startTime, increment = map(float, sys.argv[3].split('+'))
else:
    isMoveTime = True
    movetime = int(sys.argv[3])/1000
overhead = 20

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
    if(tot == scores[0]):return 1, -1000, 100
    if(tot == scores[4]):return 1, 1000, 100
    percentage = score/tot
    eloDelta = eloDiff(percentage)
    resP = results/tot
    resDev = resP*(scores-percentage)**2
    stdDeviation = sqrt(resDev.sum()) / sqrt(tot)
    low = 0
    high = 1
    for i in range(100):
        mid = (low+high)/2
        try:
            x = get_delta(percentage, mid, stdDeviation)
        except:
            low = high = 1
            break
        if x > abs(eloDelta):
            high = mid
        else:
            low = mid
    try:
        delta = get_delta(percentage, 0.95, stdDeviation)
    except:
        delta = np.nan
    return (low+high)/2, eloDelta, delta

def getLimit(wTime, bTime):
    if isMoveTime:
        return engine.Limit(time=moveTime)
    else:
        return engine.Limit(white_clock=wTime, black_clock=bTime, white_inc=increment, black_inc=increment)

def playGame(startFen, prog1, prog2):
    global startTime
    curProg, otherProg = prog1, prog2
    board = Board(startFen)
    remaindTimes = [startTime]*2
    moves = []
    termination = "Normal"
    while not board.is_game_over() and not board.can_claim_draw():
        if not isMoveTime:
            startSpan = time.time()
        result = curProg.play(board, getLimit(*remaindTimes))
        if not isMoveTime:
            endTime = time.time()
            timeSpent = endTime-startSpan
            remaindTimes[board.turn] -= timeSpent
            if remaindTimes[board.turn] < 0:
                winner = not board.turn
                termination = "Time forfeit"
                break
            remaindTimes[board.turn] += increment+overhead/1000
        board.push(result.move)
        moves.append(result.move)
        curProg, otherProg = otherProg, curProg
    moves = board.root().variation_san(moves)
    if board.can_claim_draw():
        winner = None
    elif board.outcome():
        winner = board.outcome().winner
    if winner == WHITE:
        return moves, 0, termination
    elif winner == BLACK:
        return moves, 1, termination
    else:
        return moves, 2, termination

def playBatch(args):
    id, rangeGame, globalRes = args
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
        order = [(0, prog1, prog2), (1, prog2, prog1)]
        if idBeginBoard%2 == 1:
            order[0], order[1] = order[1], order[0]
        for idProg, prog, _prog in order:
            moves, winner, termination = playGame(beginBoard, prog, _prog)
            log.write(f'[White "{sys.argv[1+idProg]}"]\n[Black "{sys.argv[2-idProg]}"]\n')
            log.write(f'[Variant "From Position"]\n[FEN "{beginBoard}"]\n')
            log.write(f'[Termination "{termination}"]\n')
            log.write(moves+'\n\n')
            log.flush()
            interResults[min(winner ^ idProg, 2)] += 1
            #print(board.outcome().winner)
            prog1.configure({'Clear Hash':None})
            prog2.configure({'Clear Hash':None})
        key = interResults[0]*2+interResults[2]
        results[key] += 1
        globalRes[key] += 1
        _, eloChange, delta = get_confidence(np.array(globalRes))
        nbL = id//10
        glob = (nbProcess+9)//10
        remaind = glob-nbL
        sys.stdout.write('\n'*nbL+'\r'+'\t'*(id%10)*2+'/'.join(map(str, results))+'\n'*remaind+'\r'+'/'.join(map(str, globalRes))+f' {eloChange:6.2f} +/- {delta:6.2f} ({sum(globalRes)})'+'\033[F'*glob+'\r')
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
with SharedMemoryManager() as smm:
    sl = smm.ShareableList([0]*5)
    pool = Pool(nbProcess)
    results = np.array(pool.map(playBatch, [(id, range(id*nbBoards//nbProcess, (id+1)*nbBoards//nbProcess), sl) for id in range(nbProcess)]))
print("\n"*((nbProcess+9)//10))
Aresults = results.sum(axis=0)
print('/'.join(map(str, Aresults)))
#thank to https://3dkingdoms.com/chess/elo.htm


confidence, eloDelta, difference = get_confidence(Aresults)
print(f"{eloDelta} +/- {difference}")
print(f"the first version is better than the second with a probability of {confidence}")
