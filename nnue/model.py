import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import time
from torch.utils.data import DataLoader, TensorDataset, random_split
from random import shuffle, seed, randrange
from tqdm import trange

class Model(nn.Module):
    inputSize = 64*12
    HLSize = 64
    SCALE = 400
    QA = 255
    QB = 64
    normal = 20
    def activation(self, x):
        return torch.clamp(x, min=0, max=self.QA)**2

    def outAct(self, x):
        return F.sigmoid(x/self.normal)

    def __init__(self, device):
        super().__init__()
        self.tohidden = nn.Linear(self.inputSize, self.HLSize)
        self.toout = nn.Linear(self.HLSize*2, 1)
        self.to(device)
        self.transfo = torch.arange(self.inputSize)^56^64
        self.device = device

    def calc_score(self, x, color, isInt=False):
        hiddenRes = torch.zeros(x.shape[0], self.HLSize*2, device=self.device)
        firstIndex = color*self.HLSize
        secondIndex = (1-color)*self.HLSize
        hiddenRes[:, firstIndex:firstIndex+self.HLSize] = self.activation(self.tohidden(x))
        hiddenRes[:, secondIndex:secondIndex+self.HLSize] = self.activation(self.tohidden(x[:, self.transfo]))
        x = self.toout(hiddenRes)
        x2 = x-self.toout.bias[0]
        if isInt:
            x2 //= self.QA
        else:
            x2 /= self.QA
        x2 += self.toout.bias[0]
        if isInt:
            return x2*self.SCALE//(self.QA*self.QB)
        else:
            return x2*self.SCALE/(self.QA*self.QB)

    def forward(self, x, color):
        return self.outAct(self.calc_score(x, color))
    
    def _round(self):
        self.toout.weight = self.toout.weight.round()
        self.toout.bias = self.toout.bias.round()
        self.tohidden.weight = self.tohidden.weight.round()
        self.tohidden.bias = self.tohidden.bias.round()

class Trainer:
    def __init__(self, lr, device):
        self.model = Model(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss(reduction="mean")
        self.device = device
        self.modelEval = Model(device)
    
    def trainStep(self, dataX, dataY, color):
        yhat = self.model(dataX, color)
        loss = self.loss_fn(dataY, yhat)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()

    def testLoss(self, dataX, dataY, color):
        yhat = self.modelEval(dataX, color)
        loss = self.loss_fn(dataY, yhat)
        return loss.item()

    def train(self, epoch, dataX, dataY, percentTrain=0.9, batchSize=100000, fileBest="bestModel.bin", testPos=None):
        startTime = time.time()
        dataset1 = TensorDataset(dataX[0], dataY[0])
        dataset2 = TensorDataset(dataX[1], dataY[1])
        dataTrain1, dataTest1 = random_split(dataset1, [percentTrain, 1-percentTrain])
        dataTrain2, dataTest2 = random_split(dataset2, [percentTrain, 1-percentTrain])
        totTrainData = len(dataTrain1)+len(dataTrain2)
        totTestData = sum(map(len, dataX))-totTrainData
        dataL1 = DataLoader(dataset=dataTrain1, batch_size=batchSize, shuffle=True)
        dataL2 = DataLoader(dataset=dataTrain2, batch_size=batchSize, shuffle=True)
        testDataL2 = DataLoader(dataset=dataTest1, batch_size=batchSize, shuffle=False)
        testDataL1 = DataLoader(dataset=dataTest2, batch_size=batchSize, shuffle=False)
        lastTestLoss = lastLoss = 0.0
        miniLoss = 1000
        isMin = False
        lastModel = Model(self.device)
        endTime = time.time()
        print(f"setup in {endTime-startTime:.5f}s")
        if testPos is not None:
            self.modelEval.load_state_dict(self.model.state_dict())
            self.modelEval.eval()
            with torch.no_grad():
                print("result of test eval before training:", self.modelEval.calc_score(testPos.float().to(self.device), 0, True)[:, 0].tolist())
        for i in range(epoch):
            startTime = time.time()
            totLoss = 0
            self.model.train()
            for c, dataL in enumerate((dataL1, dataL2)):
                for xBatch, yBatch in dataL:
                    xBatch = xBatch.float().to(self.device)
                    yBatch = yBatch.float().to(self.device)
                    totLoss += self.trainStep(xBatch, yBatch, c)*len(xBatch)
            totTestLoss = 0
            with torch.no_grad():
                self.modelEval.load_state_dict(self.model.state_dict())
                self.modelEval.eval()
                for c, testDataL in enumerate((testDataL1, testDataL2)):
                    for xBatch, yBatch in testDataL:
                        xBatch = xBatch.float().to(self.device)
                        yBatch = yBatch.float().to(self.device)
                        totTestLoss += self.testLoss(xBatch, yBatch, c)*len(xBatch)
            totLoss /= totTrainData
            totTestLoss /= totTestData
            endTime = time.time()
            if lastTestLoss == 0.0:
                lastTestLoss = totTestLoss
                lastLoss = totLoss
            print(f'epoch {i} training loss {totLoss:.5f} ({(totLoss-lastLoss)*100/lastLoss:+.2f}%) test loss {totTestLoss:.5f} ({(totTestLoss-lastTestLoss)*100/lastTestLoss:+.2f}%) in {endTime-startTime:.3f}s')
            if testPos is not None:
                with torch.no_grad():
                    print("test eval result:", self.modelEval.calc_score(testPos.float().to(self.device), 0, True)[:, 0].tolist())
            sys.stdout.flush()
            if lastTestLoss < totTestLoss and isMin:#if that goes up and if it's the minimum
                lastModel._round()
                self.save(fileBest, lastModel)#we save the model
            lastTestLoss = totTestLoss
            if totTestLoss < miniLoss:
                miniLoss = totTestLoss
                isMin = True
                lastModel.load_state_dict(self.modelEval.state_dict())
            else:
                isMin = False
            lastLoss = totLoss

    def get_int(self, tensor):
        tensor = float(tensor)
        self.maxi = max(self.maxi, tensor)
        self.mini = min(self.mini, tensor)
        self.s += tensor
        self.count += 1
        return int(round(tensor)).to_bytes(2, "little", signed=True) #if the value is not in 2 bytes (in int16_t), there is a problem

    def read_bytes(self, bytes):
        return torch.tensor(int.from_bytes(bytes, "little", signed=True), dtype=torch.float)

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
            f.write(self.get_int(model.toout.bias[0]))
        endTime = time.time()
        print(f'min {self.mini} max {self.maxi} sum {self.s:.2f} number of weights {self.count} mean {self.s/self.count:.5f} in {endTime-startTime:.3f}s')
        sys.stdout.flush()
    
    def load(self, filename):
        with open(filename, "rb") as f:
            with torch.no_grad():
                for i in range(self.model.inputSize):
                    for j in range(self.model.HLSize):
                        self.model.tohidden.weight[j][i] = self.read_bytes(f.read(2))
                for i in range(self.model.HLSize):
                    self.model.tohidden.bias[i] = self.read_bytes(f.read(2))
                for i in range(self.model.HLSize*2):
                    self.model.toout.weight[0][i] = self.read_bytes(f.read(2))
                self.model.toout.bias[0] = self.read_bytes(f.read(2))
