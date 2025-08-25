import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import time
from torch.utils.data import DataLoader, TensorDataset
from random import shuffle, seed, randrange
from tqdm import trange

class Model(nn.Module):
    inputSize = 64*12
    HLSize = 4096
    SCALE = 400
    QA = 255
    QB = 64

    def activation(self, x):
        return torch.clamp(x, min=0, max=self.QA)**2

    def outAct(self, x):
        return F.sigmoid(x/200)

    def __init__(self, device):
        super().__init__()
        self.tohidden = nn.Linear(self.inputSize, self.HLSize)
        self.toout = nn.Linear(self.HLSize*2, 1)
        self.to(device)
        self.transfo = torch.arange(self.inputSize)^56

    def forward(self, x, color):
        hidden1 = self.activation(self.tohidden(x))
        hidden2 = self.activation(self.tohidden(x[:, self.transfo]))
        if color:
            hiddenRes = torch.concatenate((hidden2, hidden1), axis=1) # for black, reverse the perspective (side to move's perspective must be on the first place)
        else:
            hiddenRes = torch.concatenate((hidden1, hidden2), axis=1)
        x = self.toout(hiddenRes)
        x2 = (x-self.toout.bias[0])/self.QA+self.toout.bias[0]
        return self.outAct(x2*self.SCALE/(self.QA*self.QB))
    
class Trainer:
    def __init__(self, lr, device):
        self.model = Model(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss(reduction="mean")
        self.device = device
    
    def trainStep(self, dataX, dataY, color):
        self.model.train()
        yhat = self.model(dataX, color)
        loss = self.loss_fn(dataY, yhat)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()

    def testLoss(self, dataX, dataY, color):
        self.model.eval()
        yhat = self.model(dataX, color)
        loss = self.loss_fn(dataY, yhat)
        return loss.item()

    def train(self, epoch, dataX, dataY, percentTrain=0.9, batchSize=100000, fileBest="bestModel.bin"):
        s = randrange(0, 2**64)
        seed(s)
        shuffle(dataX[0])
        shuffle(dataX[1])
        seed(s)
        shuffle(dataY[0])
        shuffle(dataY[1])
        sizeTrain1 = int(len(dataX[0])*percentTrain)
        sizeTrain2 = int(len(dataX[1])*percentTrain)
        totTrainData = sizeTrain1+sizeTrain2
        totTestData = sum(map(len, dataX))-totTrainData
        dataTrain1 = TensorDataset(dataX[0][:sizeTrain1], dataY[0][:sizeTrain1])
        dataTrain2 = TensorDataset(dataX[1][:sizeTrain2], dataY[1][:sizeTrain2])
        dataTest1 = TensorDataset(dataX[0][sizeTrain1:], dataY[0][sizeTrain1:])
        dataTest2 = TensorDataset(dataX[1][sizeTrain2:], dataY[1][sizeTrain2:])
        dataL1 = DataLoader(dataset=dataTrain1, batch_size=batchSize, shuffle=True)
        dataL2 = DataLoader(dataset=dataTrain2, batch_size=batchSize, shuffle=True)
        testDataL2 = DataLoader(dataset=dataTrain1, batch_size=batchSize, shuffle=True)
        testDataL1 = DataLoader(dataset=dataTrain2, batch_size=batchSize, shuffle=True)
        lastTestLoss = lastLoss = 0.0
        miniLoss = 1000
        isMin = False
        lastModel = Model(self.device)
        for i in range(epoch):
            startTime = time.time()
            totLoss = 0
            for c, dataL in enumerate((dataL1, dataL2)):
                for xBatch, yBatch in dataL:
                    xBatch = xBatch.float().to(self.device)
                    yBatch = yBatch.float().to(self.device)
                    totLoss += self.trainStep(xBatch, yBatch, c)*len(xBatch)
            totTestLoss = 0
            with torch.no_grad():
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
            sys.stdout.flush()
            if lastTestLoss < totTestLoss and isMin:#if that goes up and if it's the minimum
                self.save(fileBest, lastModel)#we save the model
            lastTestLoss = totTestLoss
            if totTestLoss < miniLoss:
                miniLoss = totTestLoss
                isMin = True
                lastModel.load_state_dict(self.model.state_dict())
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
