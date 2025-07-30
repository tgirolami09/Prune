import chess.pgn as pgn
import chess
import chess.engine

def stockfish_evaluation(board, time_limit = 0.01):
    result = engine.analyse(board, chess.engine.Limit(time=time_limit))
    return result['score']
import random
from tqdm import tqdm, trange
import sys
positions = []
stockfish_exe = sys.argv[1]
engine = chess.engine.SimpleEngine.popen_uci(stockfish_exe)
nbWantedBoards = int(sys.argv[2])
file = open('lichess_db_standard_rated_2015-08.pgn')
tq = tqdm()
openings_fen = {} #openening's name => list of (Fens, Stockfish score)
while 1:
    game = pgn.read_game(file)
    if game is None:break
    moves = game.mainline_moves()
    board = game.board()
    plyToPlay = random.randrange(16, 36, 2)
    numPlyPlayed = 0
    for move in moves:
        board.push(move)
        numPlyPlayed += 1
        if numPlyPlayed == plyToPlay:
            fen = board.fen()
    lowerFen = fen.lower()
    numPiecesInPos = sum(lowerFen.count(char) for char in "rnbq")
    if numPlyPlayed > plyToPlay+40 and numPiecesInPos >= 10:
        opening = game.headers['Opening']
        if opening not in openings_fen:
            openings_fen[opening] = []
        openings_fen[opening].append((stockfish_evaluation(chess.Board(fen)), fen))
    tq.update(1)
tq.close()
print(len(openings_fen))
nbBoardPerOpening = (nbWantedBoards-1)//len(openings_fen)+1
print(nbBoardPerOpening)
with open('beginBoards.out', 'w') as f:
    for opening, listFens in openings_fen.items():
        #print(listFens)
        listFens.sort(key=lambda x:abs(x[0].relative))
        for i in range(nbBoardPerOpening):
            if i >= len(listFens):break
            f.write(listFens[i][1]+'\n')

with open('allBeginBoards.out', 'w') as f:
    for opening, listFens in openings_fen.items():
        #print(listFens)
        listFens.sort(key=lambda x:abs(x[0].relative))
        f.write(opening + ':')
        for score, fen in listFens:
            f.write(fen+' '+str(score.relative)+';')
        f.write('\n')
engine.close()
