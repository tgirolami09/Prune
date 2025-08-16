import subprocess
import sys
from chess import Board, BLACK, WHITE, engine
from tqdm import tqdm, trange
from multiprocessing import Pool
from multiprocessing.managers import SharedMemoryManager
import time
import numpy as np
import argparse
isMoveTime = False
parser = argparse.ArgumentParser(prog="matchManager")
parser.add_argument("prog1", type=str, help="the executable of the new version")
parser.add_argument("prog2", type=str, help="the executable of the base version")
parser.add_argument("timeControl", type=str, help="the time control (60+1 for 1 minute and 1 second of increment by move, or 1000 => 1000ms by move)")
parser.add_argument('--sprt', action="store_true")
parser.add_argument('--moveOverHead', type=int, default=100, help="overhead by move (different from increment)")
parser.add_argument("--processes", '-p', type=int, default=70, help="the number of processes")
settings = parser.parse_args(sys.argv[1:])
timeControl = settings.timeControl

#seconds+seconds
if '+' in timeControl:
    startTime, increment = map(float, timeControl.split('+'))
else:
    isMoveTime = True
    movetime = int(timeControl)/1000
overhead = settings.moveOverHead
if settings.sprt:
    print('using sprt')
from math import log, sqrt, pi, erf
def eloDiff(percentage):
    return -400 * log(1 / percentage - 1, 10)

def eloScore(diff):
    return 1/(1+10**(-diff/400))

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

def getConfidenceGuess(hypothesis, score, stdDeviation):
    low = 0
    high = 1
    for i in range(100):
        mid = (low+high)/2
        x = get_delta(score, mid, stdDeviation)
        if x > hypothesis:
            high = mid
        else:
            low = mid
    return (low+high)/2

def get_confidence(results):
    scores = np.arange(5)/4
    score = (results*scores).sum()
    tot = results.sum()
    percentage = score/tot
    if(tot == scores[0]):return -1000, 100, 100, percentage
    if(tot == scores[4]):return 1000, 100, 100, percentage
    eloDelta = eloDiff(percentage)
    resP = results/tot
    resDev = resP*(scores-percentage)**2
    stdDeviation = sqrt(resDev.sum()/tot)
    try:
        delta = get_delta(percentage, 0.95, stdDeviation)
    except:
        delta = np.nan
    return eloDelta, delta, stdDeviation, percentage

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
            remaindTimes[board.turn] -= timeSpent-overhead/1000
            if remaindTimes[board.turn] < 0:
                winner = not board.turn
                termination = "Time forfeit"
                break
            remaindTimes[board.turn] += increment
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

def likelihood(hypothesis, realScore, stdDeviation):
    score = eloScore(hypothesis)
    return (1 + erf( (score - realScore) / (sqrt(2) * stdDeviation) ) ) / 2

def playBatch(args):
    id, rangeGame, globalRes = args
    file = f"games{id}.log"
    rangeGame = range(rangeGame.start, rangeGame.stop)
    log = open(file, "w")
    results = [0]*5 # (lose/lose) (lose/draw) ((lose/win) | (draw/draw)) (win/draw) (win/win)
    prog1 = engine.SimpleEngine.popen_uci(settings.prog1)
    prog2 = engine.SimpleEngine.popen_uci(settings.prog2)
    for idBeginBoard in rangeGame:
        beginBoard = beginBoards[idBeginBoard]
        beginBoard = beginBoard.replace('\n', '')
        interResults = [0, 0, 0]
        order = [(0, prog1, prog2, settings.prog1, settings.prog2), (1, prog2, prog1, settings.prog2, settings.prog1)]
        if (idBeginBoard+id)%2 == 1:
            order[0], order[1] = order[1], order[0]
        for idProg, prog, _prog, prog1Name, prog2Name in order:
            moves, winner, termination = playGame(beginBoard, prog, _prog)
            log.write(f'[White "{prog1Name}"]\n[Black "{prog2Name}"]\n')
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
        currentState = np.array(globalRes)
        eloChange, delta, stdDeviation, score = get_confidence(currentState)
        if settings.sprt:
            if currentState.sum() > 20:
                likelihood1 = likelihood(0, score, stdDeviation)
                likelihood2 = 1-likelihood(5, score, stdDeviation)
            else:
                likelihood1 = np.nan
                likelihood2 = np.nan
            if currentState.sum() > 100:# for assurance
                if likelihood1 > 0.95:
                    return np.array(results)
                elif likelihood2 > 0.95:
                    return np.array(results)
        nbL = id//10
        glob = (nbProcess+9)//10
        remaind = glob-nbL
        if settings.sprt:
            textInfo = f'{round(likelihood1, 2)} {round(likelihood2, 2)} {currentState.sum()}'
        else:
            textInfo = f'{currentState.sum()}'
        sys.stdout.write('\n'*nbL+'\r'+'\t'*(id%10)*2+'/'.join(map(str, results))+'\n'*remaind+'\r'+'/'.join(map(str, globalRes))+f' {eloChange:6.2f} +/- {delta:6.2f} ({textInfo})'+'\033[F'*glob+'\r')
        #sys.stdout.write('\r'+'\t'*id*2+str(round(get_confidence(results[0], results[2], results[1])[0], 5)))
        sys.stdout.flush()
    log.close()
    prog1.quit()
    prog2.quit()
    time.sleep(1)
    return np.array(results)

with open("beginBoards.out") as games:
    beginBoards = list(games.readlines())

nbProcess = settings.processes
nbBoards = len(beginBoards)
with SharedMemoryManager() as smm:
    sl = smm.ShareableList([0]*5)
    pool = Pool(nbProcess)
    results = np.array(list(pool.imap_unordered(playBatch, [(id, range(id*nbBoards//nbProcess, (id+1)*nbBoards//nbProcess), sl) for id in range(nbProcess)])))
print("\n"*((nbProcess+9)//10))
Aresults = results.sum(axis=0)
print('/'.join(map(str, Aresults)))
#thank to https://3dkingdoms.com/chess/elo.htm


eloDelta, difference, stdDeviation, score = get_confidence(Aresults)
print(f"{eloDelta} +/- {difference}")