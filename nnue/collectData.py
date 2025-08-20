from chess import Board, engine, WHITE, BLACK
import sys
from multiprocessing import Pool
movetime = int(sys.argv[3])/1000

def playGame(startFen, prog1, prog2):
    data1, data2 = {}, {} # fen:score
    curProg, otherProg = prog1, prog2
    curData, otherData = data1, data2
    board = Board(startFen)
    while not board.is_game_over():
        result = curProg.play(board, engine.Limit(time=movetime), info=engine.INFO_SCORE)
        score = result.info['score'].relative
        if not score.is_mate():
            curData[board.fen()] = score.score(), result.move
        elif score.mate() < 0:
            curData[board.fen()] = -100000, result.move
        else:
            curData[board.fen()] = 100000, result.move
        board.push(result.move)

        curProg, otherProg = otherProg, curProg
        curData, otherData = otherData, curData
    if board.outcome().winner == WHITE:
        return data1, 0
    elif board.outcome().winner == BLACK:
        return data2, 1
    else:
        return data1|data2, 2

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
                    f.write(f'{key}|{value[0]}|{value[1].uci()}\n')
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