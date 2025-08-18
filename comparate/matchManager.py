import subprocess
import sys
from chess import Board, BLACK, WHITE, engine, pgn
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
parser.add_argument("timeControl", type=str, help="the time control (60+1 => 1 minute and 1 second of increment by move, or 1000 => 1000ms by move)")
parser.add_argument('--sprt', action="store_true")
parser.add_argument('--moveOverHead', type=int, default=10, help="overhead by move (different from increment)")
parser.add_argument("--processes", '-p', type=int, default=70, help="the number of processes")
parser.add_argument("--hypothesis", nargs=2, type=int, default=[0, 5], help="hypothesis for sprt")
settings = parser.parse_args(sys.argv[1:])
assert settings.hypothesis[0] < settings.hypothesis[1], "the first hypothesis must be less than the hypothsesis"
timeControl = settings.timeControl

#seconds+seconds
if '+' in timeControl:
    startTime, increment = map(float, timeControl.split('+'))
else:
    isMoveTime = True
    movetime = int(timeControl)/1000
overhead = settings.moveOverHead
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

def getLimit(bTime, wTime):
    if isMoveTime:
        return engine.Limit(time=moveTime)
    else:
        return engine.Limit(white_clock=wTime, black_clock=bTime, white_inc=increment, black_inc=increment)

def playGame(startFen, prog1, prog2):
    global startTime
    curProg, otherProg = prog1, prog2
    board = Board(startFen)
    remaindTimes = [startTime]*2
    termination = "Normal"
    game = pgn.Game(dict(Variant="From Position", FEN=startFen))
    node = game
    thinkMate = False
    while not board.is_game_over() and not board.can_claim_draw():
        if not isMoveTime:
            startSpan = time.time()
        result = curProg.play(board, limit=getLimit(*remaindTimes), info=engine.INFO_ALL)
        if not isMoveTime:
            endTime = time.time()
            timeSpent = endTime-startSpan
            remaindTimes[board.turn] -= timeSpent-overhead/1000
            if remaindTimes[board.turn] < 0:
                winner = not board.turn
                termination = "Time forfeit"
                break
            remaindTimes[board.turn] += increment
        if 'score' in result.info and result.info['score'].is_mate():
            thinkMate = True
        node = node.add_variation(result.move)
        node.comment = f'[%clk {remaindTimes[board.turn]}]'
        board.push(result.move)
        curProg, otherProg = otherProg, curProg
    if board.can_claim_draw():
        winner = None
    elif board.outcome():
        winner = board.outcome().winner
    if winner == WHITE:
        return game, 0, termination, thinkMate
    elif winner == BLACK:
        return game, 1, termination, thinkMate
    else:
        return game, 2, termination, thinkMate

def likelihood(hypothesis, realScore, stdDeviation):
    score = eloScore(hypothesis)
    return (1 + erf( (score - realScore) / (sqrt(2) * stdDeviation) ) ) / 2

def playBatch(args):
    id, rangeGame, globalRes, cutoff = args
    file = f"games{id}.log"
    rangeGame = range(rangeGame.start, rangeGame.stop)
    log = open(file, "w")
    results = [0]*5 # (lose/lose) (lose/draw) ((lose/win) | (draw/draw)) (win/draw) (win/win)
    prog1 = engine.SimpleEngine.popen_uci(settings.prog1)
    prog2 = engine.SimpleEngine.popen_uci(settings.prog2)
    nbL = id//10
    glob = (nbProcess+9)//10
    remaind = glob-nbL
    for idBeginBoard in rangeGame:
        beginBoard = beginBoards[idBeginBoard]
        beginBoard = beginBoard.replace('\n', '')
        interResults = [0, 0, 0]
        order = [(0, prog1, prog2, settings.prog1, settings.prog2), (1, prog2, prog1, settings.prog2, settings.prog1)]
        if (idBeginBoard+id)%2 == 1:
            order[0], order[1] = order[1], order[0]
        for idProg, prog, _prog, prog1Name, prog2Name in order:
            game, winner, termination, thinkMate = playGame(beginBoard, prog, _prog)
            game.headers['White'] = prog1Name
            game.headers['Black'] = prog2Name
            game.headers['Result'] = ["1-0", "0-1", "1/2-1/2"][winner]
            game.headers['Termination'] = termination
            game.headers['TimeControl'] = settings.timeControl
            if winner == 2 and thinkMate:
                with open('wasntItWinning.pgn', 'a') as f:
                    f.write(str(game)+'\n\n')
            log.write(str(game)+'\n\n')
            log.flush()
            interResults[min(winner ^ idProg, 2)] += 1
            #print(board.outcome().winner)
            prog1.configure({'Clear Hash':None})
            prog2.configure({'Clear Hash':None})
            if cutoff[0]:
                break
        if cutoff[0]:
            break
        key = interResults[0]*2+interResults[2]
        results[key] += 1
        globalRes[key] += 1
        currentState = np.array(globalRes)
        eloChange, delta, stdDeviation, score = get_confidence(currentState)
        if settings.sprt:
            if currentState.sum() > 20:
                likelihood1 = likelihood(settings.hypothesis[0], score, stdDeviation)
                likelihood2 = 1-likelihood(settings.hypothesis[1], score, stdDeviation)
            else:
                likelihood1 = np.nan
                likelihood2 = np.nan
            if currentState.sum() > 100:# for assurance
                if likelihood1 > 0.95 or likelihood2 > 0.95:
                    cutoff[0] = True
                    print('\n'*glob+'\ncutoff', likelihood1, likelihood2)
                    sys.stdout.flush()
                    break
            textInfo = f'{likelihood1:.3f} {likelihood2:.3f} {currentState.sum()}'
        else:
            textInfo = f'{currentState.sum()}'
        sys.stdout.write('\n'*nbL+'\r'+'\t'*(id%10)*2+'/'.join(map(str, results))+'\n'*remaind+'\r'+'/'.join(map(str, globalRes))+f' {eloChange:6.2f} +/- {delta:6.2f} ({textInfo})'+'\033[F'*glob+'\r')
        #sys.stdout.write('\r'+'\t'*id*2+str(round(get_confidence(results[0], results[2], results[1])[0], 5)))
        sys.stdout.flush()
    log.close()
    prog1.quit()
    prog2.quit()
    return np.array(results)

with open("beginBoards.out") as games:
    beginBoards = list(games.readlines())

with open('wasntItWinning.pgn', 'w') as f:f.write("")
nbProcess = settings.processes
nbBoards = len(beginBoards)
with SharedMemoryManager() as smm:
    sl = smm.ShareableList([0]*5)
    cutoff = smm.ShareableList([False])
    pool = Pool(nbProcess)
    results = np.array(list(pool.imap_unordered(playBatch, [(id, range(id*nbBoards//nbProcess, (id+1)*nbBoards//nbProcess), sl, cutoff) for id in range(nbProcess)])))
if not cutoff[0]:
    print("\n"*((nbProcess+9)//10))
Aresults = results.sum(axis=0)
print('/'.join(map(str, Aresults)))
#thank to https://3dkingdoms.com/chess/elo.htm


eloDelta, difference, stdDeviation, score = get_confidence(Aresults)
print(f"{eloDelta} +/- {difference}")