import sys
from model import *
from chess import Board, Move, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, WHITE, BLACK
import numpy as np
from tqdm import tqdm

def reverse_mask(board):
    board = ((board&0xFFFFFFFF00000000) >> 32) | ((board&0x00000000FFFFFFFF) << 32)
    board = ((board&0xFFFF0000FFFF0000) >> 16) | ((board&0x0000FFFF0000FFFF) << 16)
    board = ((board&0xFF00FF00FF00FF00) >>  8) | ((board&0x00FF00FF00FF00FF) <<  8)
    return board

L = [np.array(tuple(map(int, bin(i)[2:].zfill(8))), dtype=np.int8) for i in range(256)]
def boardToInput(board):
    res = np.zeros(12*64, dtype=np.int8)
    for p, piece in enumerate((PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING)):
        for c, color in enumerate((WHITE, BLACK)):
            index = (p*2+c)*64
            mask = int(board.pieces(piece, color))
            for i, b in enumerate(mask.to_bytes(8, 'big')):
                eight = 8*i
                bs = L[b]
                res[index+eight:index+8+eight] = bs
    return res


import argparse
parser = argparse.ArgumentParser(prog='nnueTrainer')
parser.add_argument("dataFile", type=str, help="the filehow fast the learning go where the data is")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--device", '-d', type=str, default="cpu", help="on which device is the training")
parser.add_argument("--epoch", type=int, default=100000, help="number of epoch of training")
parser.add_argument("--batchSize", type=int, default=100000, help="batch size")
parser.add_argument("--percentTrain", type=float, default=0.9, help="percent of the dataset dedicated to training")
parser.add_argument("--reload", action="store_true", help="to not start from nothing each time")
parser.add_argument("--outFile", "-o", type=str, default="bestModel.bin", help="the file where the current best model is written")
parser.add_argument("--limit", type=int, default=-1, help="the number of training samples (-1 for all the file)")
settings = parser.parse_args(sys.argv[1:])

print('initisalise the trainer')
trainer = Trainer(settings.lr, settings.device)

if settings.reload:
    startTime = time.time()
    print("load old model")
    trainer.load(settings.outFile)
    endTime = time.time()
    print(f'in {endTime-startTime:.3f}s')

dataX = [[], []]
dataY = [[], []]
with open(settings.dataFile) as f:
    tq = tqdm()
    count = 0
    for line in f:
        assert line.count('|') == 3, line
        tq.update(1)
        fen, score, staticScore, move = line.split('|')
        score, staticScore = int(score), int(staticScore.split()[-1])
        if abs(staticScore-score) >= 70:continue
        board = Board(fen)
        dataX[not board.turn].append(boardToInput(board))
        score = 1/(1+np.exp(-float(score)/200))
        dataY[not board.turn].append([score])
        count += 1
        if count >= settings.limit and settings.limit != -1:
            break
tq.close()
print('data collected:', len(dataX[0])+len(dataX[1]))
dataX = [torch.from_numpy(np.array(i)) for i in dataX]
dataY = [torch.from_numpy(np.array(i)) for i in dataY]
print('launch training')
trainer.train(settings.epoch, dataX, dataY, settings.percentTrain, settings.batchSize, settings.outFile)
trainer.save()