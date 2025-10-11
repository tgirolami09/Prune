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
parser.add_argument("--pickledData", type=str, default="dataPickled.bin", help="where the collected data is")
parser.add_argument("--remake", action="store_true", help="to remake training data")
parser.add_argument("--fullsave", action="store_true", help="to remake training data")
parser.add_argument("--wdl", type=float, default=0.0, help="the portion of result in the target")
settings = parser.parse_args(sys.argv[1:])

print('initisalise the trainer')
trainer = Trainer(settings.lr, settings.device)

if settings.reload:
    startTime = time.time()
    print("load old model")
    trainer.load(settings.outFile)
    endTime = time.time()
    print(f'in {endTime-startTime:.3f}s')
if settings.pickledData in os.listdir() and not settings.remake:
    print("read pickled data")
    startTime = time.time()
    dataX, dataY = pickle.load(open(settings.pickledData, "rb"))
    endTime = time.time()
    totLength = len(dataX[0])+len(dataX[1])
    print(f"finished in {endTime-startTime}s with {totLength} data")
    if settings.limit != -1:
        per = int(round(len(dataX[0])*settings.limit/totLength)), int(round(len(dataX[1])*settings.limit/totLength))
        dataX[0] = dataX[0][:per[0]]
        dataX[1] = dataX[1][:per[1]]
        dataY[0] = dataY[0][:per[0]]
        dataY[1] = dataY[1][:per[1]]
        print("remaining:", len(dataX[0])+len(dataX[1]))
else:
    import re
    parts = settings.dataFile.split('/')
    directory = '/'.join(parts[:-1])
    print(directory)
    rule = re.compile(parts[-1])
    dataX = [[], []]
    dataY = [[], []]
    passed = set()
    mini = 10000
    maxi = -10000
    corrFiles = []
    for file in os.listdir(directory):
        if rule.match(file):
            corrFiles.append('/'.join((directory, file)))
    collision = 0
    filtredPos = 0
    kMaxScoreMagnitude = 1500
    kMaxMaterialImbalance = 1200
    for file in tqdm(corrFiles):
        with open(file) as f:
            count = 0
            lines = f.readlines()
            for line in tqdm(lines, leave=False):
                assert line.count('|') == 4, line
                if line in passed:
                    collision += 1
                    continue
                passed.add(line)
                fen, score, staticScore, move, result = line.split('|')
                score, staticScore = int(score), int(staticScore.split()[-1])
                if abs(score) > kMaxScoreMagnitude or abs(staticScore) > kMaxMaterialImbalance:
                    filtredPos += 1
                    continue
                board = Board(fen)
                if board.is_capture(Move.from_uci(move)) or board.is_check():
                    filtredPos += 1
                    continue
                result = float(result)
                #if abs(staticScore-score) >= 70:continue
                dataX[board.turn == BLACK].append(boardToInput(board))
                mini = min(mini, score)
                maxi = max(maxi, score)
                dataY[board.turn == BLACK].append([score, result])
                count += 1
                if count >= settings.limit and settings.limit != -1:
                    break
    print('collision', collision)
    print(f'{filtredPos*100/(filtredPos+count)}% of pos were not filtred')
    print(f'range: [{mini}, {maxi}]')
    print('data collected:', len(dataX[0])+len(dataX[1]))
    dataX = [torch.from_numpy(np.array(i)) for i in dataX]
    dataY = [torch.from_numpy(np.array(i)) for i in dataY]
    pickle.dump((dataX, dataY), open(settings.pickledData, "wb"))
print('launch training')
dataY = [trainer.model.outAct(Y[:, 0])*(1-settings.wdl)+settings.wdl*(1-Y[:, 1]) for Y in dataY]
dataY = [Y.reshape(Y.shape[0], 1) for Y in dataY]
testPos = torch.from_numpy(np.array([boardToInput(Board(fen)) for fen in [
    'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',           # starting position
    'r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w QKqk -',   # kiwipete position
    '8/8/2K5/2Q5/8/8/8/3k4 w - - 0 1',                                    # one queen advantage
    '8/8/2K5/2QQ4/8/8/8/3k4 w - - 0 1'                                    # two queen advantage
]]))
trainer.train(settings.epoch, dataX, dataY, settings.percentTrain, settings.batchSize, settings.outFile, testPos, settings.fullsave)
trainer.save()