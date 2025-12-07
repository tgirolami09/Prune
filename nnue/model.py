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
from pyzipmanager import *

class PickledTensorDataset(Dataset):
    def __init__(self, directory, wdl, act):
        self.wdl = wdl
        self.act = act
        self.directory = directory
        self.file_names = os.listdir(directory)
        self.cum = [0]*(len(self.file_names)+1)
        self.buffers = [None]*len(self.file_names)
        for i, file in enumerate(tqdm(self.file_names)):
            self.buffers[i] = zip_open((self.directory+"/"+file).encode(), 0, byref(error))
            nbdata = zip_get_num_entries(self.buffers[i], 0)
            self.cum[i+1] = self.cum[i]+nbdata
        self.num_samples = self.cum[-1]
        print(self.num_samples)

    def read_file(self, id, idx):
        c = bytes([0]*(12*64*2+2*4))
        index = zip_name_locate(self.buffers[id], id_to_name(idx), 0)
        f = zip_fopen_index(self.buffers[id], index, 0)
        num_read = zip_fread(f, c, len(c))
        zip_fclose(f)
        x, y = np.frombuffer(c[:-8], dtype=np.int8), np.frombuffer(c[-8:], dtype=np.float32)
        return x, y

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        fileId = -1
        for i in range(len(self.cum)):
            if self.cum[i] > idx:
                fileId = i-1
                break
        x, y = self.read_file(fileId, idx-self.cum[fileId])
        y = self.act(torch.tensor(y[0]))*(1-self.wdl)+self.wdl*y[1]
        return torch.from_numpy(x), y

class Model(nn.Module):
    inputSize = 64*12
    HLSize = 64
    SCALE = 400
    QA = 255
    QB = 64
    normal = 200
    def activation(self, x):
        return torch.clamp(x, min=0, max=self.QA)**2

    def outAct(self, x):
        return F.sigmoid(x/self.normal)

    def __init__(self, device):
        super().__init__()
        self.tohidden = nn.Linear(self.inputSize, self.HLSize)
        self.toout = nn.Linear(self.HLSize*2, 1, bias=False)
        self.endBias = nn.Parameter(torch.randn(1))
        self.to(device)
        self.device = device

    def calc_score(self, x, isInt=False):
        hiddenRes = torch.zeros(x.shape[0], self.HLSize*2, device=self.device)
        hiddenRes[:, :self.HLSize] = self.activation(self.tohidden(x[:, :self.inputSize]))
        hiddenRes[:, self.HLSize:] = self.activation(self.tohidden(x[:, self.inputSize:]))
        x = self.toout(hiddenRes)
        if isInt:
            x //= self.QA
        else:
            x /= self.QA
        x += self.endBias
        if isInt:
            return x*self.SCALE//(self.QA*self.QB)
        else:
            return x*self.SCALE/(self.QA*self.QB)

    def forward(self, x):
        return self.outAct(self.calc_score(x))
    
    def _round(self):
        self.toout.weight[:] = self.toout.weight.round()
        self.endBias[:] = self.endBias.round()
        self.tohidden.weight[:] = self.tohidden.weight.round()
        self.tohidden.bias[:] = self.tohidden.bias.round()

class Clipper:
    clamp = 127
    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clamp(-self.clamp, self.clamp)
            module.weight.data = w
        if hasattr(module, 'bias') and module.bias is not None:
            b = module.bias.data
            b = b.clamp(-self.clamp, self.clamp)
            module.bias.data = b
        elif hasattr(module, 'endBias'):
            b = module.endBias.data
            b = b.clamp(-self.clamp, self.clamp)
            module.endBias.data = b

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
                print("result of test eval before training:", self.modelEval.calc_score(testPos.float().to(self.device), True)[:, 0].tolist())
        clipper = Clipper()
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
            self.model.apply(clipper)
            with torch.no_grad():
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
                    print("test eval result:", self.modelEval.calc_score(testPos.float().to(self.device), True)[:, 0].tolist())
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

    def get_int(self, tensor):
        tensor = float(tensor)
        self.maxi = max(self.maxi, abs(tensor))
        return int(round(tensor)).to_bytes(1, sys.byteorder, signed=True) #if the value is not in 2 bytes (in int16_t), there is a problem

    def read_bytes(self, bytes):
        return torch.tensor(int.from_bytes(bytes, sys.byteorder, signed=True), dtype=torch.float)

    def save(self, filename="model.txt", model=None):
        if model is None:
            model = self.model
        startTime = time.time()
        self.maxi = -1000
        self.mini =  1000
        self.s = 0
        self.count = 0
        with open(filename, "wb") as f:
            for i in range(model.inputSize):
                for j in range(model.HLSize):
                    f.write(self.get_int(model.tohidden.weight[j][i]))
            for i in range(model.HLSize):
                f.write(self.get_int(model.tohidden.bias[i]))
            for i in range(model.HLSize*2):
                f.write(self.get_int(model.toout.weight[0][i]))
            f.write(self.get_int(model.endBias[0]))
        endTime = time.time()
        print(f'save model to {filename} (max |weight| {self.maxi}) in {endTime-startTime:.3f}s')
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
