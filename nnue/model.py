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

tranform = np.arange(12*64)^(56^64)

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

    def read(self, idx):
        for i in range(len(self.cum)):
            if self.cum[i] > idx:break
        i -= 1
        p = self.games[i]
        X = np.frombuffer(self.raw[p:p+12*64], dtype=np.int8).copy()
        p += 12*64
        Y = [int.from_bytes(self.raw[p:p+3], signed=True), int.from_bytes(self.raw[p+3:p+4])]
        p += 4
        p += 2
        for t in range(idx-self.cum[i]):
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
            Y = [int.from_bytes(self.raw[p:p+3], signed=True), int.from_bytes(self.raw[p+3:p+4])]
            p += 4
        Y[1], color = divmod(Y[1], 2)
        if color == 1:
            X = np.concatenate((X[tranform], X))
            Y[0] *= -1
            Y[1] = 2-Y[1]
        else:
            X = np.concatenate((X, X[tranform]))
        return X, Y


class PickledTensorDataset(Dataset):
    def __init__(self, directory, wdl, act):
        self.wdl = wdl
        self.act = act
        self.directory = directory
        self.file_names = os.listdir(directory)
        self.cum = [0]*(len(self.file_names)+1)
        self.buffers = [myDeflate(directory+"/"+name) for name in self.file_names]
        for i, file in enumerate(tqdm(self.file_names)):
            nbdata = len(self.buffers[i])
            self.cum[i+1] = self.cum[i]+nbdata
        self.num_samples = self.cum[-1]
        print(self.num_samples)

    def read_file(self, id, idx):
        return self.buffers[id].read(idx)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        fileId = -1
        for i in range(len(self.cum)):
            if self.cum[i] > idx:
                fileId = i-1
                break
        x, y = self.read_file(fileId, idx-self.cum[fileId])
        yn = self.act(torch.tensor(y[0]))*(1-self.wdl)+self.wdl*y[1]/2
        assert not yn.isnan(), y
        return torch.from_numpy(x.copy()), torch.from_numpy(np.array([yn]))

def roundQ(tensor, Q):
    return (tensor*Q).round()/Q

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

    def train(self, epoch, directory, percentTrain=0.9, batchSize=100000, fileBest="bestModel.bin", testPos=None, processes=1, wdl=0.0):
        startTime = time.time()
        dataset = PickledTensorDataset(directory, wdl, self.model.outAct)
        dataTrain, dataTest = random_split(dataset, [percentTrain, 1-percentTrain])
        totTrainData = len(dataTrain)
        totTestData = len(dataset)-totTrainData
        dataL = DataLoader(dataset=dataTrain, batch_size=batchSize, shuffle=True, num_workers=processes)
        testDataL = DataLoader(dataset=dataTest, batch_size=batchSize, shuffle=False, num_workers=processes)
        lastTestLoss = lastLoss = 0.0
        miniLoss = 1000
        current_lr = self.lr
        endTime = time.time()
        print(f"setup in {endTime-startTime:.5f}s")
        if testPos is not None:
            self.modelEval.load_state_dict(self.model.state_dict())
            self.modelEval.eval()
            with torch.no_grad():
                self.modelEval._round()
                print("result of test eval before training:", self.modelEval.get_cp(testPos.float().to(self.device))[:, 0].tolist())
        for i in range(epoch):
            startTime = time.time()
            totLoss = 0
            self.model.train()
            for xBatch, yBatch in dataL:
                xBatch = xBatch.float().to(self.device)
                yBatch = yBatch.float().to(self.device)
                totLoss += self.trainStep(xBatch, yBatch)*len(xBatch)
            totTestLoss = 0
            endTimeTrain = time.time()
            with torch.no_grad():
                self.model.clamp()
                self.modelEval.load_state_dict(self.model.state_dict())
                self.modelEval._round()
                self.modelEval.eval()
                for xBatch, yBatch in testDataL:
                    xBatch = xBatch.float().to(self.device)
                    yBatch = yBatch.float().to(self.device)
                    totTestLoss += self.testLoss(xBatch, yBatch)*len(xBatch)
            totLoss /= totTrainData
            totTestLoss /= totTestData
            endTime = time.time()
            if lastTestLoss == 0.0:
                lastTestLoss = totTestLoss
                lastLoss = totLoss
            span = endTime-startTime
            span2 = endTimeTrain-startTime
            print(f'epoch {i} training loss {totLoss:.5f} ({(totLoss-lastLoss)*100/lastLoss:+.2f}%) test loss {totTestLoss:.5f} ({(totTestLoss-lastTestLoss)*100/lastTestLoss:+.2f}%) in {span:.3f}s ({span2/span*100:.2f}% for training) lr {current_lr}')
            if testPos is not None:
                with torch.no_grad():
                    print("test eval result:", self.modelEval.get_cp(testPos.float().to(self.device))[:, 0].tolist())
            sys.stdout.flush()
            if totTestLoss < miniLoss:
                miniLoss = totTestLoss
                with torch.no_grad():
                    self.save(fileBest, self.modelEval)
            elif totTestLoss > lastTestLoss:
                current_lr /= 10**.5
                for g in self.optimizer.param_groups:
                    g['lr'] = current_lr
            lastTestLoss = totTestLoss
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
                        self.model.tohidden.weight[j][i] = self.read_bytes(f.read(1))
                for i in range(self.model.HLSize):
                    self.model.tohidden.bias[i] = self.read_bytes(f.read(1))
                for i in range(self.model.HLSize*2):
                    self.model.toout.weight[0][i] = self.read_bytes(f.read(1))
                self.model.endBias[0] = self.read_bytes(f.read(1))
