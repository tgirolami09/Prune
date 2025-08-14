import sys
import subprocess
from chess import pgn, engine
from datetime import date
import chess
import time
startpos = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'

prog1 = engine.SimpleEngine.popen_uci(sys.argv[1])
prog2 = engine.SimpleEngine.popen_uci(sys.argv[2])
nodes1 = nodes2 = 0
# Start time in minutes+increment in s (like 5+3 for 5 minutes and 3 seconds of increment by move)
# or movetime in s
timeControl = sys.argv[3]
isMoveTime = False
if '+' in timeControl:
    startTime, increment = map(float, timeControl.split('+'))
else:
    isMoveTime = True
    moveTime = float(timeControl)

def getLimit():
    if isMoveTime:
        return engine.Limit(time=moveTime)
    else:
        return engine.Limit(white_clock=wTime, black_clock=bTime, white_inc=increment, black_inc=increment)

def updateLimit(timeSpent, side):
    if isMoveTime:return False
    global wTime, bTime
    #first move is not counted in time spent
    if hasPlayedFirst[side]:
        if side == chess.WHITE:
            wTime -= timeSpent
            wTime += increment+additionalIncrement/1000
            timeRemaind = wTime
            color = 'White'
        else:
            bTime -= timeSpent
            bTime += increment+additionalIncrement/1000
            timeRemaind = bTime
            color = 'Black'
        if timeRemaind > 0:
            print(f"Prog {game.headers[color]} has {timeRemaind}s left")
        else:
            print(f"Player {game.headers['Black']} has no time left")
            winnerByTime = chess.WHITE
            return True
    else:
        hasPlayedFirst[side] = True
    return False

if not isMoveTime:
    #Time in s
    wTime = bTime = startTime * 60
    #small increment in ms to reduce effect of fetching data
    additionalIncrement = 20
if len(sys.argv) > 4:
    board = chess.Board(sys.argv[4] if sys.argv[4] != 'startpos' else startpos)
    if(len(sys.argv) > 6):
        sideLimit = int(sys.argv[5])
        EloLimit = int(sys.argv[6])
        if sideLimit:prog = prog2
        else: prog = prog1
        prog.configure({'UCI_limitStrength':True})
        prog.configure({'UCI_Elo':EloLimit})
else:
    board = chess.Board()
startFen = board.fen()
if startFen != startpos:
    game = pgn.Game(dict(Variant="From Position", FEN=startFen))
else:
    game = pgn.Game()
node = game
moves = []
player1 = sys.argv[1]
player2 = sys.argv[2]
game.headers["White"] = sys.argv[2-board.turn]
game.headers["Black"] = sys.argv[board.turn+1]
game.headers['TimeControl'] = str(startTime*60)+'+'+str(increment)
winnerByTime = -1
hasPlayedFirst = [False, False]
while not board.is_game_over():
    print(board.fen())
    print(' '.join(i.uci() for i in board.move_stack))
    startTime = time.time()
    result = prog1.play(board, limit=getLimit(), info=engine.INFO_ALL)
    endTime = time.time()
    timeSpent = endTime-startTime
    rmove = result.move
    print(result.info)
    #Because no nodes are visited when a book move was found
    nbNodes = result.info.get('nodes', 0)
    nodes1 += nbNodes
    if updateLimit(timeSpent, board.turn):
        break

    board.push(rmove)
    node = node.add_variation(rmove)
    print(board)
    prog1, prog2 = prog2, prog1
    nodes1, nodes2 = nodes2, nodes1
    player1, player2 = player2, player1
#Won by time
game.headers['Event'] = "Test Game"
if board.outcome() == None:
    winner = winnerByTime
    game.headers["Termination"] = "Time forfeit"
else:
    winner = board.outcome().winner
    game.headers["Termination"] = "Normal"
game.headers["Date"] = str(date.today()).replace('-', '.')
if winner == chess.WHITE:
    print(sys.argv[1])
    game.headers['Result'] = '1-0'
elif winner == chess.BLACK:
    print(sys.argv[2])
    game.headers['Result'] = '0-1'
else:
    print('draw')
    game.headers['Result'] = '1/2-1/2'
print(game)
print(player1, nodes1)
print(player2, nodes2)
prog1.quit()
prog2.quit()