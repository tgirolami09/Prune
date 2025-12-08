import sys
from model import *
from chess import Board, Move, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, WHITE, BLACK
import numpy as np
from tqdm import tqdm
import os
import pickle
def reverse_mask(board):
    board = ((board&0xFFFFFFFF00000000) >> 32) | ((board&0x00000000FFFFFFFF) << 32)
    board = ((board&0xFFFF0000FFFF0000) >> 16) | ((board&0x0000FFFF0000FFFF) << 16)
    board = ((board&0xFF00FF00FF00FF00) >>  8) | ((board&0x00FF00FF00FF00FF) <<  8)
    return board

L = [np.array(tuple(map(int, bin(i)[2:].zfill(8))), dtype=np.int8) for i in range(256)]
def boardToInput(board):
    res = np.zeros(12*64, dtype=np.int8)
    index = 0
    for p, piece in enumerate((board.pawns, board.knights, board.bishops, board.rooks, board.queens, board.kings)):
        for c, color in enumerate((WHITE, BLACK)):
            mask = piece&board.occupied_co[color]
            for i, b in enumerate(mask.to_bytes(8, 'big')):
                eight = 8*i
                bs = L[b]
                res[index+eight:index+8+eight] = bs
            index += 64
    return res

def fullInput(board):
    res = np.zeros(12*64*2, dtype=np.int8)
    if board.turn == BLACK:
        res[12*64:] = boardToInput(board)
        res[:12*64] = boardToInput(board.mirror())
    else:
        res[:12*64] = boardToInput(board)
        res[12*64:] = boardToInput(board.mirror())
    return res

import argparse
parser = argparse.ArgumentParser(prog='nnueTrainer')
parser.add_argument("pickledData", type=str, default="dataPickled.bin", help="where the collected data is")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--device", '-d', type=str, default="cpu", help="on which device is the training")
parser.add_argument("--epoch", type=int, default=100000, help="number of epoch of training")
parser.add_argument("--batchSize", type=int, default=100000, help="batch size")
parser.add_argument("--percentTrain", type=float, default=0.9, help="percent of the dataset dedicated to training")
parser.add_argument("--reload", action="store_true", help="to not start from nothing each time")
parser.add_argument("--outFile", "-o", type=str, default="bestModel.bin", help="the file where the current best model is written")
parser.add_argument("--limit", type=int, default=-1, help="the number of training samples (-1 for all the file)")
parser.add_argument("--remake", action="store_true", help="to remake training data")
parser.add_argument("--wdl", type=float, default=0.0, help="the portion of result in the target")
parser.add_argument("--processes", "-p", type=int, default=1, help="number of processes used for unpacking data")
parser.add_argument("--nbSuperBatch", "-s", type=int, default=1, help="number of superbatch")

settings = parser.parse_args(sys.argv[1:])

print('initisalise the trainer')
trainer = Trainer(settings.lr, settings.device)

if settings.reload:
    startTime = time.time()
    print("load old model")
    trainer.load(settings.outFile)
    endTime = time.time()
    print(f'in {endTime-startTime:.3f}s')
print('launch training')
testPos = torch.from_numpy(np.array([fullInput(Board(fen)) for fen in [
    'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',           # starting position
    'r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w QKqk -',   # kiwipete position
    '8/8/2K5/2Q5/8/8/8/3k4 w - - 0 1',                                    # one queen advantage
    '8/8/2K5/2QQ4/8/8/8/3k4 w - - 0 1'                                    # two queen advantage
]]))
trainer.train(settings.epoch, settings.pickledData, settings.percentTrain, settings.batchSize, settings.outFile, testPos, settings.processes, settings.wdl, settings.nbSuperBatch)
trainer.save()