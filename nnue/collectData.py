from chess import Board, engine, WHITE, BLACK
import sys, math, time
from multiprocessing import Pool
from multiprocessing.managers import SharedMemoryManager
import asyncio
import argparse
import os
isMoveTime = False
parser = argparse.ArgumentParser(prog="matchManager")
parser.add_argument("prog1", type=str, help="the executable of one of the version")
parser.add_argument("prog2", type=str, help="the executable of another or the same version")
parser.add_argument("timeControl", type=str, help="the time control (60+1 => 1 minute and 1 second of increment by move, or 1000 => 1000ms by move)")
parser.add_argument('--moveOverHead', type=int, default=10, help="overhead by move (different from increment)")
parser.add_argument("--processes", '-p', type=int, default=70, help="the number of processes")
parser.add_argument("--file", type=str, default="beginBoards.out", help="the file where the straing boards are")
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

class EvalResult:
    ev = 0
    def __init__(self, ev:int):
        self.ev = ev

    def __repr__(self):
        return f'static eval: {self.ev}'

async def staticEval(protocol, board):
    class UciStaticEval(engine.BaseCommand[EvalResult]):
        def __init__(self, prot:engine.UciProtocol) -> None:
            super().__init__(prot)
            self.engine = prot

        def start(self) -> None:
            self.engine._position(board)
            self.engine.send_line('isready')

        def line_received(self, line:str) -> None:
            if line.startswith('static evaluation:'):
                self.result.set_result(int(line.split()[2]))
                self.set_finished()
            elif line.strip() == "readyok":
                self.engine.send_line('eval')
    return await protocol.communicate(UciStaticEval)

def getStaticEval(motor, board):
    with motor._not_shut_down():
        coro = asyncio.wait_for(staticEval(motor.protocol, board), motor.timeout)
        future = asyncio.run_coroutine_threadsafe(coro, motor.protocol.loop)
    return future.result()

def playGame(startFen, prog1, prog2):
    global startTime
    data1, data2 = {}, {}
    curProg, otherProg = prog1, prog2
    curData, otherData = data1, data2 #fen:(score, move)
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
                curData[board.fen()] = (str(score.score()), str(getStaticEval(curProg, board)), result.move)
        board.push(result.move)
        curProg, otherProg = otherProg, curProg
        curData, otherData = otherData, curData
    winner = None
    if board.outcome() is not None:
        winner = board.outcome().winner
    if winner == WHITE:
        return data1, data2, 0
    elif winner == BLACK:
        return data2, data1, 1
    else:
        return data1, data2, 2

def renderTime(s):
    res = ''
    if s > 60:
        m = int(s//60)
        s -= 60*m
        if m > 60:
            h, m = divmod(m, 60)
            if h > 24:
                d, h = divmod(h, 24)
                res += f'{d:2d}d '
            res += f'{h:2d}h '
        res += f'{m:2d}m '
    res += f'{s:5.2f}s'
    return res

def playBatch(args):
    id, rangeGame, cumul = args
    results = [0, 0, 0]
    with open(f'data{id}.out', "w") as f:f.write('')
    prog1 = engine.SimpleEngine.popen_uci(sys.argv[1])
    prog2 = engine.SimpleEngine.popen_uci(sys.argv[2])
    total = len(beginBoards)
    startGen = time.time()
    for idBeginBoard in rangeGame:
        beginBoard = beginBoards[idBeginBoard]
        beginBoard = beginBoard.replace('\n', '')
        for idProg, prog, _prog in ((0, prog1, prog2), (1, prog2, prog1)):
            data1, data2, result = playGame(beginBoard, prog, _prog)
            results[min(result^idProg, 2)] += 1
            score = 1 if result != 2 else 0.5
            with open(f'data{id}.out', "a") as f:
                for key, value in data1.items():
                    f.write(f'{key}|{value[0]}|{value[1]}|{value[2].uci()}|{score}\n')
                score = 1-score
                for key, value in data2.items():
                    f.write(f'{key}|{value[0]}|{value[1]}|{value[2].uci()}|{score}\n')
        cumul[0] += 1
        columns = os.get_terminal_size().columns
        elapsedTime = time.time()-startGen
        if elapsedTime > cumul[0]:
            unit = 's/it'
            speed = elapsedTime/cumul[0]
            remainingTime = (total-cumul[0])*speed
        else:
            unit = 'it/s'
            speed = cumul[0]/elapsedTime
            remainingTime = (total-cumul[0])*elapsedTime/cumul[0]
        string = f'{cumul[0]*100/total:4.1f}% {speed:.2f}{unit} {renderTime(remainingTime)} ['
        percent = cumul[0]*(columns-len(string)-1)/total
        string += '█'*int(percent)
        string += chr(9615-int(percent*7)%7)
        string += (columns-1-len(string)) * ' ' + ']\r'
        sys.stdout.write(string)
        sys.stdout.flush()
    prog1.quit()
    prog2.quit()
    return results

nbProcess = 70
with open(settings.file) as games:
    beginBoards = list(games.readlines())
nbBoards = len(beginBoards)
pool = Pool(nbProcess)
with SharedMemoryManager() as smm:
    sl = smm.ShareableList([0])
    results = pool.map(playBatch, [(id, range(id*nbBoards//nbProcess, (id+1)*nbBoards//nbProcess), sl) for id in range(nbProcess)])

print("\n"*((nbProcess+9)//10))
wins = 0
loses = 0
draws = 0
for result in results:
    wins += result[0]
    loses += result[1]
    draws += result[2]
print(f"\nwins = {wins}, draws = {draws}, loses = {loses}")