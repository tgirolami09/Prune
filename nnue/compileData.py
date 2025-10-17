from tqdm import tqdm
from chess import Board, BLACK, WHITE, Move, PAWN
import sys
import argparse
import re
import numpy as np
import pickle
import os
import torch
pieces = ['p', 'n', 'b', 'r', 'q', 'k']
def col(square):
    return square&7

def row(square):
    return square >> 3

def to_uci(pos):
    uci = chr((7-col(pos))+ord('a'))
    uci += chr(row(pos)+ord('1'))
    return uci

def reverse_mask(board):
    board = ((board&0xFFFFFFFF00000000) >> 32) | ((board&0x00000000FFFFFFFF) << 32)
    board = ((board&0xFFFF0000FFFF0000) >> 16) | ((board&0x0000FFFF0000FFFF) << 16)
    board = ((board&0xFF00FF00FF00FF00) >>  8) | ((board&0x00FF00FF00FF00FF) <<  8)
    return board

def mirror(board):
    board = ((board&0xF0F0F0F0F0F0F0F0) >> 4) | ((board&0x0F0F0F0F0F0F0F0F) << 4)
    board = ((board&0xCCCCCCCCCCCCCCCC) >> 2) | ((board&0x3333333333333333) << 2)
    board = ((board&0xAAAAAAAAAAAAAAAA) >> 1) | ((board&0x5555555555555555) << 1)
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

def uciFromInfo(moveInfo):
    start = (moveInfo >> 6) & 0x3f
    end = moveInfo & 0x3f
    promot = moveInfo >> 12
    mv = to_uci(start)+to_uci(end)
    if promot != -1:
        mv += pieces[promot]
    return mv

def readGame(file):
    global filtredPos, mini, maxi, count
    board = Board.empty()
    board.castling_rights = Board().castling_rights
    for p, piece in enumerate(("pawns", "knights", "bishops", "rooks", "queens", "kings")):
        for c, color in enumerate((WHITE, BLACK)):
            bitboard = mirror(int.from_bytes(file.read(8), sys.byteorder))
            setattr(board, piece, bitboard|getattr(board, piece))
            board.occupied |= bitboard
            board.occupied_co[color] |= bitboard
    gameInfo = file.read(1)[0]
    board.turn = BLACK if gameInfo%2 else WHITE
    result = gameInfo//2/2
    sizeGame = int.from_bytes(file.read(2), sys.byteorder, signed=True)
    for i in range(sizeGame):
        doStore = not bool(f.read(1)[0])
        moveInfo = int.from_bytes(f.read(2), sys.byteorder, signed=True)
        nextMove = Move.from_uci(uciFromInfo(moveInfo))
        score = int.from_bytes(f.read(4), sys.byteorder, signed=True)
        staticScore = int.from_bytes(f.read(4), sys.byteorder, signed=True)
        if doStore:
            if abs(staticScore) > kMaxMaterialImbalance or abs(score) > kMaxScoreMagnitude:
                filtredPos += 1
            elif board.is_capture(nextMove) or board.is_check():
                filtredPos += 1
            else:
                #if abs(staticScore-score) >= 70:continue
                dataX[board.turn == BLACK].append(boardToInput(board))
                mini = min(mini, score)
                maxi = max(maxi, score)
                dataY[board.turn == BLACK].append([score, result])
                count += 1
        result = 1-result
        if i == 0 and board.piece_type_at(nextMove.from_square) == PAWN and abs(nextMove.from_square-nextMove.to_square)%8 != 0 and board.piece_type_at(nextMove.to_square) is None:
            board.ep_square = nextMove.to_square
        board.push(nextMove)


parser = argparse.ArgumentParser(prog='nnueTrainer')
parser.add_argument("dataFile", type=str, help="the file where the data is")
parser.add_argument("pickledData", type=str, help="where the training data will be")
settings = parser.parse_args(sys.argv[1:])

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
    with open(file, "rb") as f:
        count = 0
        tq = tqdm(leave=False)
        while f.tell() != os.fstat(f.fileno()).st_size:
            readGame(f)
            tq.update(1)
        tq.close()
print('collision', collision)
print(f'{filtredPos*100/(filtredPos+count)}% of pos were not filtred')
print(f'range: [{mini}, {maxi}]')
print('data collected:', len(dataX[0])+len(dataX[1]))
dataX = [torch.from_numpy(np.array(i)) for i in dataX]
dataY = [torch.from_numpy(np.array(i)) for i in dataY]
pickle.dump((dataX, dataY), open(settings.pickledData, "wb"))