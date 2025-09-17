import subprocess
import sys
from chess import Board, BLACK, WHITE, engine, pgn
from tqdm import tqdm, trange
from multiprocessing import Pool
from multiprocessing.managers import SharedMemoryManager
import time
import numpy as np
import argparse
import math
import os
from scipy import optimize
np.seterr(divide='raise')
isMoveTime = False
parser = argparse.ArgumentParser(prog="matchManager")
parser.add_argument("prog1", type=str, help="the executable of the new version")
parser.add_argument("prog2", type=str, help="the executable of the base version")
parser.add_argument("timeControl", type=str, help="the time control (60+1 => 1 minute and 1 second of increment by move, or 1000 => 1000ms by move)")
parser.add_argument('--sprt', action="store_true")
parser.add_argument('--moveOverHead', type=int, default=10, help="overhead by move (different from increment)")
parser.add_argument("--processes", '-p', type=int, default=70, help="the number of processes")
parser.add_argument("--hypothesis", nargs=2, type=int, default=[0, 5], help="hypothesis for sprt")
parser.add_argument("--confidence", nargs=1, type=float, default=0.95, help="confidence of the bounds for sprt")
parser.add_argument("--configs", nargs=2, type=eval, default=[{}, {}], help="config of the different engines")
parser.add_argument("--openingFile", type=str, default="beginBoards.out", help="file where are the starting fens")
settings = parser.parse_args(sys.argv[1:])
assert isinstance(settings.configs[0], dict) and isinstance(settings.configs[1], dict), f"configs must be a dict {settings.configs}"
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
    if board.turn == BLACK:
        curProg, otherProg = otherProg, curProg
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

#stole the code from OpenBench https://github.com/AndyGrant/OpenBench/blob/877f7e8302d7c854447e762cc92c1afe0c53940c/OpenBench/stats.py
def stats(pdf):
    epsilon = 1e-6
    for i in pdf:
        assert -epsilon <= i[1] <= 1 + epsilon
    n = sum([prob for value, prob in pdf])
    assert abs(n - 1) < epsilon
    s = sum([prob * value for value, prob in pdf])
    var = sum([prob * (value - s) ** 2 for value, prob in pdf])
    return s, var


def uniform(pdf):
    n = len(pdf)
    return [(ai, 1 / n) for ai, pi in pdf]


def secular(pdf):
    """
    Solves the secular equation sum_i pi*ai/(1+x*ai)=0.
    """
    epsilon = 1e-9
    v, w = pdf[0][0], pdf[-1][0]
    values = [ai for ai, pi in pdf]
    v = min(values)
    w = max(values)
    assert v * w < 0
    l = -1 / w
    u = -1 / v

    def f(x):
        return sum([pi * ai / (1 + x * ai) for ai, pi in pdf])

    x, res = optimize.brentq(
        f, l + epsilon, u - epsilon, full_output=True, disp=False
    )
    assert res.converged
    return x


def MLE_tvalue(pdfhat, ref, s):

    N = len(pdfhat)
    pdf_MLE = uniform(pdfhat)
    for i in range(10):
        pdf_ = pdf_MLE
        mu, var = stats(pdf_MLE)
        sigma = var ** (1 / 2)
        pdf1 = [
            (ai - ref - s * sigma * (1 + ((mu - ai) / sigma) ** 2) / 2, pi)
            for ai, pi in pdfhat
        ]
        x = secular(pdf1)
        pdf_MLE = [
            (pdfhat[i][0], pdfhat[i][1] / (1 + x * pdf1[i][0])) for i in range(N)
        ]
        if max([abs(pdf_[i][1] - pdf_MLE[i][1]) for i in range(N)]) < 1e-9:
            break

    return pdf_MLE


def PentanomialSPRT(results, elo0, elo1):

    ## Implements https://hardy.uhasselt.be/Fishtest/normalized_elo_practical.pdf

    # Ensure no division by 0 issues
    results = [max(1e-3, x) for x in results]

    # Partial computation of Normalized t-value
    nelo_divided_by_nt = 800 / math.log(10)
    nt0, nt1 = (x / nelo_divided_by_nt for x in (elo0, elo1))
    t0, t1 = nt0 * math.sqrt(2), nt1 * math.sqrt(2)

    # Number of game-pairs, and the PDF of Ptnml(0-2) expressed as (0-1)
    N = sum(results)
    pdf = [(i / 4, results[i] / N) for i in range(0, 5)]

    # Pdf given each normalized t-value, and then the LLR process for each
    pdf0, pdf1 = (MLE_tvalue(pdf, 0.5, t) for t in (t0, t1))
    mle_pdf    = [(math.log(pdf1[i][1]) - math.log(pdf0[i][1]), pdf[i][1]) for i in range(len(pdf))]

    return N * stats(mle_pdf)[0]

def playBatch(args):
    id, rangeGame, globalRes, cutoff = args
    file = f"games{id}.log"
    rangeGame = range(rangeGame.start, rangeGame.stop)
    log = open(file, "w")
    results = [0]*5 # (lose/lose) (lose/draw) ((lose/win) | (draw/draw)) (win/draw) (win/win)
    prog1 = engine.SimpleEngine.popen_uci(settings.prog1)
    prog2 = engine.SimpleEngine.popen_uci(settings.prog2)
    prog1.configure(settings.configs[0])
    prog2.configure(settings.configs[1])
    boundUp = math.log(settings.confidence/(1-settings.confidence))
    boundDown = -boundUp
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
            #prog1.configure({'Clear Hash':None})
            #prog2.configure({'Clear Hash':None})
            if cutoff[0]:
                break
        if cutoff[0]:
            break
        key = interResults[0]*2+interResults[2]
        results[key] += 1
        globalRes[key] += 1
        currentState = np.array(globalRes)
        try:
            eloChange, delta, stdDeviation, score = get_confidence(currentState)
        except:
            eloChange, delta, stdDeviation, score = (np.nan,)*4
        if settings.sprt:
            try:
                llr = PentanomialSPRT(currentState, *settings.hypothesis)
            except:
                llr = np.nan
            if llr > boundUp or llr < boundDown or (llr is np.nan and currentState.sum() > 50):
                cutoff[0] = True
                print('\ncutoff', llr, boundDown, boundUp)
                sys.stdout.flush()
                break
            textInfo = f'{llr:.3f} ({boundDown}, {boundUp}) {currentState.sum()}'
        else:
            textInfo = f'{currentState.sum()}'
        string = '/'.join(map(str, globalRes))+f' {eloChange:6.2f} +/- {delta:6.2f} ({textInfo})['
        columns = os.get_terminal_size().columns
        totLength = columns-len(string)-1
        percent = currentState.sum()*totLength/len(beginBoards)
        string += '█'*int(percent)
        string += chr(9615-int(percent*7)%7)
        string += (columns-1-len(string))*' '+']\r'
        sys.stdout.write(string)
        #sys.stdout.write('\r'+'\t'*id*2+str(round(get_confidence(results[0], results[2], results[1])[0], 5)))
        sys.stdout.flush()
    log.close()
    prog1.quit()
    prog2.quit()
    return np.array(results)

with open(settings.openingFile) as games:
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