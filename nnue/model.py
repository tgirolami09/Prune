import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
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
        return F.sigmoid(x)

    def __init__(self, device):
        super().__init__()
        self.tohidden = nn.Linear(self.inputSize, self.HLSize)
        self.toout = nn.Linear(self.HLSize*2, 1)
        self.to(device)
    
    def forward(self, x, color):
        hidden1 = self.activation(self.tohidden(x[:, :self.inputSize]))
        hidden2 = self.activation(self.tohidden(x[:, self.inputSize:]))
        if color:
            hiddenRes = torch.concatenate((hidden2, hidden1), axis=1) # for black, reverse the perspective (side to move's perspective must be on the first place)
        else:
            hiddenRes = torch.concatenate((hidden1, hidden2), axis=1)
        return self.outAct(self.toout(hiddenRes))
    
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

    def train(self, epoch, dataX, dataY, percentTrain=0.9, batchSize=100000):
        s = randrange(0, 2**64)
        seed(s)
        shuffle(dataX[0])
        shuffle(dataX[1])
        seed(s)
        shuffle(dataY[0])
        shuffle(dataY[1])
        sizeTrain1 = int(len(dataX[0])*percentTrain)
        sizeTrain2 = int(len(dataX[1])*percentTrain)
        dataTrain1 = TensorDataset(dataX[0][:sizeTrain1], dataY[0][:sizeTrain1])
        dataTrain2 = TensorDataset(dataX[1][:sizeTrain2], dataY[1][:sizeTrain2])
        dataTest1 = TensorDataset(dataX[0][sizeTrain1:], dataY[0][sizeTrain1:])
        dataTest2 = TensorDataset(dataX[1][sizeTrain2:], dataY[1][sizeTrain2:])
        dataL1 = DataLoader(dataset=dataTrain1, batch_size=batchSize, shuffle=True)
        dataL2 = DataLoader(dataset=dataTrain2, batch_size=batchSize, shuffle=True)
        testDataL2 = DataLoader(dataset=dataTrain1, batch_size=batchSize, shuffle=True)
        testDataL1 = DataLoader(dataset=dataTrain2, batch_size=batchSize, shuffle=True)
        lastTestLoss = lastLoss = 0.0
        for i in range(epoch):
            totLoss = 0
            for c, dataL in enumerate((dataL1, dataL2)):
                for xBatch, yBatch in dataL:
                    xBatch = xBatch.to(self.device)
                    yBatch = yBatch.to(self.device)
                    totLoss += self.trainStep(xBatch, yBatch, c)
            totTestLoss = 0
            with torch.no_grad():
                for c, testDataL in enumerate((testDataL1, testDataL2)):
                    for xBatch, yBatch in testDataL:
                        xBatch = xBatch.to(self.device)
                        yBatch = yBatch.to(self.device)
                        totTestLoss += self.testLoss(xBatch, yBatch, c)
            print(f'\repoch {i} training loss {totLoss:.5f} ({lastLoss:.5f}) test loss {totTestLoss:.5f} ({lastTestLoss:.5f})', end='')
            sys.stdout.flush()
            lastTestLoss = totTestLoss
            lastLoss = totLoss

    def save(self, filename="model.txt"):
        with open(filename) as f:
            for i in range(self.model.inputSize):
                for j in range(self.model.HLSize):
                    f.write(str(self.model.tohidden.weights[i][j])+' ')
                f.write('\n')
            for i in range(self.model.HLSize):
                f.write(str(self.model.tohidden.biases[i])+' ')
            f.write('\n')
            for i in range(self.model.HLSize*2):
                f.write(str(self.model.toout.weights[i][0])+' ')
            f.write('\n')
            f.write(str(self.model.toout.biases[0])+'\n')