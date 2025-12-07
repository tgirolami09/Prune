from tqdm import tqdm
from chess import Board, BLACK, WHITE, Move, PAWN
from multiprocessing import Pool
import sys
import argparse
import re
import numpy as np
import pickle
import os
from pyzipmanager import *
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

def moveFromInfo(moveInfo):
    start = ((moveInfo >> 6) & 0x3f)^7
    end = (moveInfo & 0x3f)^7
    promot = moveInfo >> 12
    if promot == -1:
        promot = None
    else:
        promot += 1
    return Move(start, end, promot)

def readGame(file, fw, idMove):
    count = 0
    filtredPos = 0
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
    if board.turn == BLACK:
        result = 1-result
    sizeGame = int.from_bytes(file.read(2), sys.byteorder, signed=True)
    dataX = np.zeros(12*2*64, dtype=np.int8)
    dataY = np.zeros(2, dtype=np.float32)
    for i in range(sizeGame):
        doStore = not bool(file.read(1)[0])
        moveInfo = int.from_bytes(file.read(2), sys.byteorder, signed=True)
        nextMove = moveFromInfo(moveInfo)
        score = int.from_bytes(file.read(4), sys.byteorder, signed=True)
        staticScore = int.from_bytes(file.read(4), sys.byteorder, signed=True)
        if doStore:
            if abs(staticScore) > kMaxMaterialImbalance or abs(score) > kMaxScoreMagnitude:
                filtredPos += 1
            elif board.is_capture(nextMove) or board.is_check():
                filtredPos += 1
            else:
                #if abs(staticScore-score) >= 70:continue
                pv, npv = board, board.mirror()
                if(board.turn == BLACK):
                    pv, npv = npv, pv
                dataX = np.zeros(12*2*64, np.int8)
                dataX[:12*64] = boardToInput(pv)
                dataX[12*64:] = boardToInput(npv)
                dataY[0] = score
                dataY[1] = result
                name = hex(idMove)
                data = dataX.tobytes()+dataY.tobytes()
                source = zip_source_buffer(fw, data, len(data), 0)
                zip_file_add(fw, name.encode(), source, ZIP_FL_OVERWRITE)
                idMove += 1
                count += 1
        result = 1-result
        if i == 0 and board.piece_type_at(nextMove.from_square) == PAWN and abs(nextMove.from_square-nextMove.to_square)%8 != 0 and board.piece_type_at(nextMove.to_square) is None:
            board.ep_square = nextMove.to_square
        board.push(nextMove)
    return count, filtredPos, idMove

def count_games(name):
    nbGame = 0
    nbMove = 0
    with open(name, "rb") as f:
        f.seek(0, 2)
        size_file = f.tell()
        f.seek(0, 0)
        while f.tell() != size_file:
            f.seek(8*2*6+1, 1)
            sizeGame = int.from_bytes(f.read(2), sys.byteorder, signed=True)
            nbGame += 1
            for i in range(sizeGame):
                nbMove += not bool(f.read(1)[0])
                f.seek(10, 1)
    return nbGame, nbMove


def readFile(arg):
    id, name = arg
    count = 0
    filtredPos = 0
    nbGame, nbMoves = count_games(name)
    print(nbGame, nbMoves)
    filename = settings.pickledData+"/data"+str(id)+".zip"
    idMove = 0
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(name, "rb") as f:
        for i in range(nbGame):
            z = zip_open(filename.encode(), ZIP_CREATE, byref(error))
            a, b, idMove = readGame(f, z, idMove)
            count += a
            filtredPos += b
            print(i, "/", nbGame)
            zip_close(z)
    return count, filtredPos

parser = argparse.ArgumentParser(prog='nnueTrainer')
parser.add_argument("dataFile", type=str, help="the file where the data is")
parser.add_argument("pickledData", type=str, help="where the training data will be")
parser.add_argument("--processes", "-p", type=int, default=1, help="number of processes to process the data")
settings = parser.parse_args(sys.argv[1:])

parts = settings.dataFile.split('/')
directory = '/'.join(parts[:-1])
rule = re.compile(parts[-1])
dataX = np.array([], dtype=np.int8)
dataY = np.array([], dtype=np.float16)
corrFiles = []
for file in os.listdir(directory):
    if rule.match(file):
        corrFiles.append('/'.join((directory, file)))
collision = 0
filtredPos = 0
kMaxScoreMagnitude = 1500
kMaxMaterialImbalance = 1200
count = 0
with Pool(settings.processes) as p:
    for c, fP in tqdm(p.imap_unordered(readFile, list(enumerate(corrFiles))), total=len(corrFiles)):
        count += c
        filtredPos += fP

print(f'{filtredPos*100/(filtredPos+count)}% of pos were filtred')
print('data collected:', count)