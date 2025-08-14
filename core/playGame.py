import sys
import subprocess
from chess import pgn, engine
import chess
import time
startpos = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'

prog1 = engine.SimpleEngine.popen_uci(sys.argv[1])
prog2 = engine.SimpleEngine.popen_uci(sys.argv[2])
nodes1 = nodes2 = 0
#Start time in minutes
startTime = float(sys.argv[3])
#Time in s
wTime = bTime = startTime * 60
#Increment in s 
increment = int(sys.argv[4]) 
#small increment in ms to reduce effect of fetching data
additionalIncrement = 20
if len(sys.argv) > 5:
    board = chess.Board(sys.argv[5] if sys.argv[5] != 'startpos' else startpos)
    if(len(sys.argv) > 7):
        sideLimit = int(sys.argv[6])
        EloLimit = int(sys.argv[7])
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
game.headers["White"] = sys.argv[1]
game.headers["Black"] = sys.argv[2]
winnerByTime = -1
hasPlayedFirst = [False, False]
while not board.is_game_over() and not board.is_seventyfive_moves():
    print(board.fen())
    print(' '.join(i.uci() for i in board.move_stack))
    timeSpent = -time.time()
    result = prog1.play(board, limit=engine.Limit(white_clock=wTime, black_clock=bTime, white_inc=increment, black_inc=increment), info=engine.INFO_ALL)
    timeSpent +=time.time()
    rmove = result.move
    #Because no nodes are visited when a book move was found
    try:
        nbNodes = result.info['nodes']
        nodes1 += nbNodes
    except:
        print("Could not get the nb of nodes")

    if (board.turn == chess.WHITE):
        if (hasPlayedFirst[0]):
            wTime -= timeSpent
            if (wTime > 0):
                wTime += increment + additionalIncrement/1000
                print(f"Prog {game.headers['White']} has {wTime}s left")
            else:
                print(f"Player {game.headers['White']} has no time left")
                winnerByTime = chess.BLACK
                break
        else:
            hasPlayedFirst[0] = True
    else:
        if (hasPlayedFirst[1]):
            bTime -= timeSpent
            if (bTime > 0):
                bTime += increment + additionalIncrement/1000
                print(f"Prog {game.headers['Black']} has {bTime}s left")
            else:
                print(f"Player {game.headers['Black']} has no time left")
                winnerByTime = chess.WHITE
                break
        else:
            hasPlayedFirst[1] = True

    board.push(rmove)
    node = node.add_variation(rmove)
    print(board)
    prog1, prog2 = prog2, prog1
    nodes1, nodes2 = nodes2, nodes1
    player1, player2 = player2, player1
print(game)
#Won by time
if board.outcome() == None:
    winner = winnerByTime
else:
    winner = board.outcome().winner

if winner == chess.WHITE:
    print(sys.argv[1])
elif winner == chess.BLACK:
    print(sys.argv[2])
else:
    print('draw')
print(player1, nodes1)
print(player2, nodes2)
prog1.quit()
prog2.quit()