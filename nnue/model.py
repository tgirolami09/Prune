from multiprocessing import Pool
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import time, os
import pickle
import numpy as np
import io
from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset
from random import shuffle, seed, randrange
from tqdm import trange, tqdm

transform = np.arange(12*64)^(56^64)

def compress(X):
    return np.packbits(X, axis=-1, bitorder="little")

intToArr = [np.array(tuple(map(int, bin(i)[2:].zfill(8))), dtype=np.int8) for i in range(256)]

def uncompress(X):
    res = np.unpackbits(X, bitorder="little")
    return np.float32(np.concatenate((res, res[transform])))


class myDeflate:
    def __init__(self, name):
        self.name = name
        with open(name, "rb") as f:
            self.raw = f.read()
        self.games = []
        self.cum = [0]
        i = 0
        while i < len(self.raw):
            self.games.append(i)
            i += 12*64
            i += 4
            nb = int.from_bytes(self.raw[i:i+2])
            i += 2
            for k in range(nb):
                a = self.raw[i]
                b = self.raw[i+1]
                n = a+b
                i += 2+4
                i += ((3**a).bit_length()+7 >> 3) + ((3**b).bit_length()+7 >> 3) + a+b
            self.cum.append(self.cum[-1]+nb+1)
    
    def __len__(self):
        return self.cum[-1]

    def lower(self, idx):
        low, high = 1, len(self.cum)
        while low < high:
            mid = (low+high)//2
            if self.cum[mid] < idx:
                low = mid+1
            elif self.cum[mid] > idx:
                high = mid
            else:
                return mid-1
        return low-1

    def read_range(self, start, end, formula):
        resX = np.zeros((self.cum[end]-self.cum[start], 12*64//8), dtype=np.uint8)
        resY = np.zeros((self.cum[end]-self.cum[start], 1), dtype=np.float32)
        idData = 0
        for i in range(start, end):
            p = self.games[i]
            X = np.frombuffer(self.raw[p:p+12*64], dtype=np.int8).copy()
            p += 12*64
            resX[idData] = compress(X)
            resY[idData] = formula(torch.tensor(int.from_bytes(self.raw[p:p+3], signed=True), dtype=torch.float32), (self.raw[p+3]//2)/2)
            idData += 1
            p += 4
            N = int.from_bytes(self.raw[p:p+2])
            p += 2
            for t in range(N):
                a, b = self.raw[p:p+2]
                p += 2
                for s, n in ((1, a), (-1, b)):
                    indexes = np.array(list(self.raw[p:p+n]), dtype=np.int32)
                    p += n
                    n2 = (3**n).bit_length()+7 >> 3
                    T = int.from_bytes(self.raw[p:p+n2])
                    for r in range(n-1, -1, -1):
                        indexes[r] += 64*4*(T%3)
                        T //= 3
                    X[indexes] += s
                    p += n2
                Y = formula(torch.tensor(int.from_bytes(self.raw[p:p+3], signed=True), dtype=torch.float32), (self.raw[p+3]//2)/2)
                p += 4
                resX[idData] = compress(X)
                resY[idData] = Y
                idData += 1
        return resX, resY

class SuperBatch:
    def __init__(self, directory, wdl, nbSuperBatch, nbProcess):
        self.wdl = wdl
        self.directory = directory
        self.file_names = os.listdir(directory)
        self.cum = [0]*(len(self.file_names)+1)
        self.buffers = [myDeflate(directory+"/"+name) for name in self.file_names]
        for i, file in enumerate(tqdm(self.file_names)):
            nbdata = len(self.buffers[i])
            self.cum[i+1] = self.cum[i]+nbdata
        self.num_samples = self.cum[-1]
        print(self.num_samples)
        self.nbSuperBatch = nbSuperBatch
        self.processes = nbProcess

    def __len__(self):
        return self.num_samples

    def find_index(self, idx):
        low, high = 1, len(self.cum)
        while low < high:
            mid = (low+high)//2
            #print(mid, self.cum[mid], idx)
            if self.cum[mid] < idx:
                low = mid+1
            elif self.cum[mid] > idx:
                high = mid
            else:
                return mid-1
        return low-1

    def launch_worker(self, args):
        id, start, end = args
        return self.buffers[id].read_range(start, end, lambda a, b:(1-self.wdl)*outAct(a)+self.wdl*b)

    def __getitem__(self, idx):
        tot = len(self)
        start = idx*tot//self.nbSuperBatch
        end = (idx+1)*tot//self.nbSuperBatch
        startFile = self.find_index(start)
        realStart = self.buffers[startFile].lower(start-self.cum[startFile])
        endFile = self.find_index(end)
        realEnd = self.buffers[endFile].lower(end-self.cum[endFile])
        sbData = self.buffers[endFile].cum[realEnd]-self.buffers[startFile].cum[realStart]+self.buffers[startFile].cum[-1]
        sbData += self.cum[endFile]-self.cum[startFile+1]
        resX = np.zeros((sbData, 64*12//8), dtype=np.uint8)
        resY = np.zeros((sbData, 1), dtype=np.float32)
        idData = 0
        with Pool(self.processes) as p:
            for curX, curY in p.imap_unordered(self.launch_worker, [
                (i, realStart*(i == startFile), len(self.buffers[i].cum)-1 if i != endFile else realEnd)
                for i in range(startFile, endFile+1)
            ]):
                resX[idData:idData+len(curX)] = curX
                resY[idData:idData+len(curY)] = curY
                idData += len(curX)
        assert(idData == len(resX))
        return resX, resY

class CompressedBatch(Dataset):
    def __init__(self, dataX, dataY):
        self.dataX, self.dataY = dataX, dataY
    
    def __len__(self):
        return len(self.dataX)
    
    def __getitem__(self, idx):
        return torch.from_numpy(uncompress(self.dataX[idx])), self.dataY[idx]

def roundQ(tensor, Q):
    return (tensor*Q).round()/Q

def outAct(x):
    return F.sigmoid(x/Model.normal)

class Model(nn.Module):
    inputSize = 64*12
    HLSize = 64
    SCALE = 400
    QA = 255
    QB = 64
    normal = 200
    def activation(self, x):
        return torch.clamp(x, min=0, max=1)**2

    def outAct(self, x):
        return F.sigmoid(x/self.normal)

    def __init__(self, device):
        super().__init__()
        self.tohidden = nn.Linear(self.inputSize, self.HLSize)
        self.toout = nn.Linear(self.HLSize*2, 1)
        self.to(device)
        self.device = device

    def calc_score(self, x):
        hiddenRes = torch.zeros(x.shape[0], self.HLSize*2, device=self.device)
        hiddenRes[:, :self.HLSize] = self.activation(self.tohidden(x[:, :self.inputSize]))
        hiddenRes[:, self.HLSize:] = self.activation(self.tohidden(x[:, self.inputSize:]))
        x = self.toout(hiddenRes)
        return x

    def forward(self, x):
        return F.sigmoid(self.calc_score(x))
    
    def get_cp(self, x):
        return torch.floor(self.calc_score(x)*self.SCALE)

    def _round(self):
        self.toout.weight[:] = roundQ(self.toout.weight, self.QB)
        self.toout.bias[:] = roundQ(self.toout.bias, self.QB*self.QA)
        self.tohidden.weight[:] = roundQ(self.tohidden.weight, self.QA)
        self.tohidden.bias[:] = roundQ(self.tohidden.bias, self.QA)

    def clamp(self):
        clampA = 127/self.QA
        clampB = 127/self.QB
        clampC = 32767/(self.QA*self.QB)
        self.toout.weight[:] = torch.clamp(self.toout.weight, -clampB, clampB)
        self.toout.bias[:] = torch.clamp(self.toout.bias, -clampC, clampC)
        self.tohidden.weight[:] = torch.clamp(self.tohidden.weight, -clampA, clampA)
        self.tohidden.bias[:] = torch.clamp(self.tohidden.bias, -clampA, clampA)

class Trainer:
    def __init__(self, lr, device):
        self.model = Model(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss(reduction="mean")
        self.device = device
        self.lr = lr
        self.modelEval = Model(device)
    
    def trainStep(self, dataX, dataY):
        yhat = self.model(dataX)
        loss = self.loss_fn(dataY, yhat)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()

    def testLoss(self, dataX, dataY):
        yhat = self.modelEval(dataX)
        loss = self.loss_fn(dataY, yhat)
        return loss.item()

    def train(self, epoch, directory, percentTrain, batchSize, fileBest, testPos, processes, wdl, nbSuperBatch):
        startTime = time.time()
        SB = SuperBatch(directory, wdl, nbSuperBatch, processes)
        lastLoss = 0.0
        current_lr = self.lr
        totTrainData = len(SB)
        endTime = time.time()
        print(f"setup in {endTime-startTime:.5f}s")
        if testPos is not None:
            self.modelEval.load_state_dict(self.model.state_dict())
            self.modelEval.eval()
            with torch.no_grad():
                self.modelEval._round()
                print("result of test eval before training:", self.modelEval.get_cp(testPos.float().to(self.device))[:, 0].tolist())
        for idEpoch in range(epoch):
            startTime = time.time()
            totLoss = 0
            self.model.train()
            for idSB in trange(nbSuperBatch, leave=False):
                Batch = CompressedBatch(*SB[idSB])
                dataL = DataLoader(Batch, batch_size=batchSize, shuffle=True, num_workers=processes, pin_memory=self.device=="cuda")
                for xBatch, yBatch in tqdm(dataL, leave=False):
                    xBatch = xBatch.float().to(self.device)
                    yBatch = yBatch.float().to(self.device)
                    totLoss += self.trainStep(xBatch, yBatch)*len(xBatch)
            endTime = time.time()
            with torch.no_grad():
                self.model.clamp()
                self.modelEval.load_state_dict(self.model.state_dict())
                self.modelEval._round()
            totLoss /= totTrainData
            if lastLoss == 0.0:
                lastLoss = totLoss
            span = endTime-startTime
            print(f'epoch {idEpoch} training loss {totLoss:.5f} ({(totLoss-lastLoss)*100/lastLoss:+.2f}%) in {span:.3f}s lr {current_lr}')
            if testPos is not None:
                with torch.no_grad():
                    print("test eval result:", self.modelEval.get_cp(testPos.float().to(self.device))[:, 0].tolist())
            sys.stdout.flush()
            with torch.no_grad():
                self.save("modelEpoch"+str(idEpoch)+".bin", self.modelEval)
            current_lr *= 0.99
            for g in self.optimizer.param_groups:
                g['lr'] = current_lr
            lastLoss = totLoss
        
        for g in self.optimizer.param_groups:
            g['lr'] = self.lr

    def get_int(self, tensor, nbytes=1):
        tensor = float(tensor)
        return int(round(tensor)).to_bytes(nbytes, sys.byteorder, signed=True) #if the value is not in 2 bytes (in int16_t), there is a problem

    def read_bytes(self, bytes):
        return torch.tensor(int.from_bytes(bytes, sys.byteorder, signed=True), dtype=torch.float)

    def save(self, filename="model.txt", model=None):
        if model is None:
            model = self.model
        startTime = time.time()
        with open(filename, "wb") as f:
            for i in range(model.inputSize):
                for j in range(model.HLSize):
                    f.write(self.get_int(model.tohidden.weight[j][i]*model.QA))
            for i in range(model.HLSize):
                f.write(self.get_int(model.tohidden.bias[i]*model.QA))
            for i in range(model.HLSize*2):
                f.write(self.get_int(model.toout.weight[0][i]*model.QB))
            f.write(self.get_int(model.toout.bias[0]*model.QA*model.QB, 2))
        endTime = time.time()
        print(f'save model to {filename} in {endTime-startTime:.3f}s')
        sys.stdout.flush()
    
    def load(self, filename):
        with open(filename, "rb") as f:
            with torch.no_grad():
                for i in range(self.model.inputSize):
                    for j in range(self.model.HLSize):
                        self.model.tohidden.weight[j][i] = self.read_bytes(f.read(1))/self.model.QA
                for i in range(self.model.HLSize):
                    self.model.tohidden.bias[i] = self.read_bytes(f.read(1))/self.model.QA
                for i in range(self.model.HLSize*2):
                    self.model.toout.weight[0][i] = self.read_bytes(f.read(1))/self.model.QB
                self.model.toout.bias[0] = self.read_bytes(f.read(2))/(self.QA*self.QB)
