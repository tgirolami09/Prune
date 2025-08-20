import sys
from model import *
from chess import Board, Move, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, WHITE, BLACK
import numpy as np
from tqdm import tqdm

def reverse_mask(board):
    board = (board&0xFFFFFFFF00000000) >> 32 | (board&0x00000000FFFFFFFF) << 32
    board = (board&0xFFFF0000FFFF0000) >> 16 | (board&0x0000FFFF0000FFFF) << 16
    board = (board&0xFF00FF00FF00FF00) >>  8 | (board&0x00FF00FF00FF00FF) <<  8
    return board

def boardToInput(board):
    res = np.zeros(12*64*2)
    for p, piece in enumerate((PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING)):
        for c, color in enumerate((WHITE, BLACK)):
            index = (p*2+c)*64*2
            mask = int(board.pieces(piece, color))
            res[index   :index+64 ] = np.array(list(map(int, bin(mask              )[2:].zfill(64))))
            res[index+64:index+128] = np.array(list(map(int, bin(reverse_mask(mask))[2:].zfill(64))))#yes, I like when same caracter are on the same column
    return res


import argparse
parser = argparse.ArgumentParser(prog='nnueTrainer')
parser.add_argument("dataFile", type=str, help="the filehow fast the learning go where the data is")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--device", '-d', type=str, default="cpu", help="on which device is the training")
parser.add_argument("--epoch", type=int, default=100000, help="number of epoch of training")
parser.add_argument("--batchSize", type=int, default=100000, help="batch size")
parser.add_argument("--percentTrain", type=float, default=0.9, help="percent of the dataset dedicated to training")
settings = parser.parse_args(sys.argv[1:])

dataX = [[], []]
dataY = [[], []]
with open(settings.dataFile) as f:
    tq = tqdm()
    for line in f:
        assert line.count('|') == 2, line
        fen, score, move = line.split('|')
        board = Board(fen)
        board.push(Move.from_uci(move.strip()))
        dataX[not board.turn].append(boardToInput(board))
        dataY[not board.turn].append([float(score)])
        tq.update(1)
tq.close()
print('data collected')
dataX = [torch.from_numpy(np.array(i)).float() for i in dataX]
dataY = [torch.from_numpy(np.array(i)).float() for i in dataY]
print('initisalise the trainer')
trainer = Trainer(settings.lr, settings.device)
print('launch training')
trainer.train(settings.epoch, dataX, dataY, settings.percentTrain, settings.batchSize)
trainer.save()