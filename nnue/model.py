import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
        self.tohidden = nn.Linear(self.inputSize, self.HLSize*2)
        self.toout = nn.Linear(self.HLSize*2, 1)
        self.to(device)
    
    def forward(self, x, color):
        hidden = self.activation(self.tohidden(x))
        if color:
            hidden = torch.concatenate((hidden[:self.HLSize], hidden[self.HLSize:]))
        return self.outAct(self.toout(hidden))
    
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
            print(f'epoch {i} training loss {totLoss} test loss {totTestLoss}')