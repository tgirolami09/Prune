from chess import Board, engine, WHITE, BLACK
import sys, math, time
from multiprocessing import Pool
import argparse
isMoveTime = False
parser = argparse.ArgumentParser(prog="matchManager")
parser.add_argument("prog1", type=str, help="the executable of one of the version")
parser.add_argument("prog2", type=str, help="the executable of another or the same version")
parser.add_argument("timeControl", type=str, help="the time control (60+1 => 1 minute and 1 second of increment by move, or 1000 => 1000ms by move)")
parser.add_argument('--moveOverHead', type=int, default=10, help="overhead by move (different from increment)")
parser.add_argument("--processes", '-p', type=int, default=70, help="the number of processes")
settings = parser.parse_args(sys.argv[1:])
timeControl = settings.timeControl

#seconds+seconds
if '+' in timeControl:
    startTime, increment = map(float, timeControl.split('+'))
else:
    isMoveTime = True
    movetime = int(timeControl)/1000
    startTime = 0
overhead = settings.moveOverHead

def getLimit(bTime, wTime):
    if isMoveTime:
        return engine.Limit(time=moveTime)
    else:
        return engine.Limit(white_clock=wTime, black_clock=bTime, white_inc=increment, black_inc=increment)

def playGame(startFen, prog1, prog2):
    global startTime
    curProg, otherProg = prog1, prog2
    data = {} #fen:(score, move)
    board = Board(startFen)
    remaindTimes = [startTime]*2
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
                break
            remaindTimes[board.turn] += increment
        if 'score' in result.info:
            score = result.info['score'].relative
            if not score.is_mate():
                data[board.fen()] = (str(score.score()), str(curProg.staticEval()), result.move)
        board.push(result.move)
        curProg, otherProg = otherProg, curProg
    winner = None
    if board.outcome() is not None:
        winner = board.outcome().winner
    if winner == WHITE:
        return data, 0
    elif winner == BLACK:
        return data, 1
    else:
        return data, 2

def playBatch(args):
    id, rangeGame = args
    results = [0, 0, 0]
    with open(f'data{id}.out', "w") as f:f.write('')
    prog1 = engine.SimpleEngine.popen_uci(sys.argv[1])
    prog2 = engine.SimpleEngine.popen_uci(sys.argv[2])
    for idBeginBoard in rangeGame:
        beginBoard = beginBoards[idBeginBoard]
        beginBoard = beginBoard.replace('\n', '')
        for idProg, prog, _prog in ((0, prog1, prog2), (1, prog2, prog1)):
            data, result = playGame(beginBoard, prog, _prog)
            results[min(result^idProg, 2)] += 1
            with open(f'data{id}.out', "a") as f:
                for key, value in data.items():
                    f.write(f'{key}|{value[0]}|{value[1]}|{value[2].uci()}\n')
        sys.stdout.write('\n'*(id//10)+'\r'+'\t'*(id%10)*2+'/'.join(map(str, (results[0], results[2], results[1])))+'\033[F'*(id//10)+'\r')
    prog1.quit()
    prog2.quit()
    return results

nbProcess = 70
with open("beginBoards.out") as games:
    beginBoards = list(games.readlines())
nbBoards = len(beginBoards)
pool = Pool(nbProcess)
results = pool.map(playBatch, [(id, range(id*nbBoards//nbProcess, (id+1)*nbBoards//nbProcess)) for id in range(nbProcess)])

print("\n"*((nbProcess+9)//10))
wins = 0
loses = 0
draws = 0
for result in results:
    wins += result[0]
    loses += result[1]
    draws += result[2]
print(f"\nwins = {wins}, draws = {draws}, loses = {loses}")