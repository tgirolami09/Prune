import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import time
from torch.utils.data import DataLoader, TensorDataset, random_split
from random import shuffle, seed, randrange
from tqdm import trange

def roundQ(tensor, Q):
    return (tensor*Q).round()/Q

class Model(nn.Module):
    inputSize = 64*12
    HLSize = 64
    SCALE = 400
    QA = 255
    QB = 64
    BUCKET = 8
    DIVISOR = (31+BUCKET)//BUCKET
    def activation(self, x):
        return torch.clamp(x, min=0, max=1)**2

    def outAct(self, x):
        return F.sigmoid(x/self.SCALE)

    def __init__(self, device):
        super().__init__()
        self.tohidden = nn.Linear(self.inputSize, self.HLSize)
        self.toout = nn.Linear(self.HLSize*2, self.BUCKET)
        self.to(device)
        self.transfo = torch.arange(self.inputSize)^56^64
        self.device = device

    def calc_score(self, x, color, isInt=False):
        hiddenRes = torch.empty(x.shape[0], self.HLSize*2, device=self.device)
        firstIndex = color*self.HLSize
        secondIndex = (1-color)*self.HLSize
        hiddenRes[:, firstIndex:firstIndex+self.HLSize] = self.tohidden(x)
        hiddenRes[:, secondIndex:secondIndex+self.HLSize] = self.tohidden(x[:, self.transfo])
        y = self.toout(self.activation(hiddenRes)).gather(1, ((x.count_nonzero(axis=1)-2)//self.DIVISOR).reshape(-1, 1))
        return y
    
    def get_static_eval(self, x, color):
        return (self.calc_score(x, color)*self.SCALE).to(torch.int)

    def forward(self, x, color):
        return F.sigmoid(self.calc_score(x, color))
    
    def _round(self):
        self.toout.weight[:] = roundQ(self.toout.weight, self.QB)
        self.toout.bias[:] = roundQ(self.toout.bias, self.QB)
        self.tohidden.weight[:] = roundQ(self.tohidden.weight, self.QA)
        self.tohidden.bias[:] = roundQ(self.tohidden.bias, self.QA)

class Clipper:
    def __call__(self, module):
        if hasattr(module, 'toout'):
            clamp = 127/module.QB
            module.toout.weight.data = module.toout.weight.data.clamp(-clamp, clamp)
            module.toout.bias.data = module.toout.bias.data.clamp(-clamp, clamp)
            clamp = 127/module.QA
            module.tohidden.weight.data = module.tohidden.weight.data.clamp(-clamp, clamp)
            module.tohidden.bias.data = module.tohidden.bias.data.clamp(-clamp, clamp)

class Trainer:
    def __init__(self, lr, device):
        self.model = torch.compile(Model(device))
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss(reduction="mean")
        self.device = device
        self.lr = lr
        self.modelEval = torch.compile(Model(device))
    
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

    def train(self, epoch, dataX, dataY, percentTrain=0.9, batchSize=100000, fileBest="bestModel.bin", testPos=None, processes=1):
        startTime = time.time()
        dataset1 = TensorDataset(dataX[0], dataY[0])
        dataset2 = TensorDataset(dataX[1], dataY[1])
        dataTrain1, dataTest1 = random_split(dataset1, [percentTrain, 1-percentTrain])
        dataTrain2, dataTest2 = random_split(dataset2, [percentTrain, 1-percentTrain])
        totTrainData = len(dataTrain1)+len(dataTrain2)
        totTestData = sum(map(len, dataX))-totTrainData
        dataL1 = DataLoader(dataset=dataTrain1, batch_size=batchSize, shuffle=True, num_workers=processes)
        dataL2 = DataLoader(dataset=dataTrain2, batch_size=batchSize, shuffle=True, num_workers=processes)
        testDataL1 = DataLoader(dataset=dataTest1, batch_size=batchSize, shuffle=False, num_workers=processes)
        testDataL2 = DataLoader(dataset=dataTest2, batch_size=batchSize, shuffle=False, num_workers=processes)
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
                print("result of test eval before training:", self.modelEval.get_static_eval(testPos.float().to(self.device), 0)[:, 0].tolist())
        clipper = Clipper()
        for i in range(epoch):
            startTime = time.time()
            totLoss = 0
            self.model.train()
            colors = {0:iter(dataL1), 1:iter(dataL2)}
            while colors:
                for color in list(colors.keys()):
                    try:
                        xBatch, yBatch = next(colors[color])
                    except StopIteration:
                        colors.pop(color)
                        continue
                    except:
                        print(colors, color)
                        raise StopIteration()
                    xBatch = xBatch.float().to(self.device)
                    yBatch = yBatch.float().to(self.device)
                    totLoss += self.trainStep(xBatch, yBatch, color)*len(xBatch)
            totTestLoss = 0
            endTimeTrain = time.time()
            self.model.apply(clipper)
            with torch.no_grad():
                self.modelEval.load_state_dict(self.model.state_dict())
                self.modelEval._round()
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
            span = endTime-startTime
            span2 = endTimeTrain-startTime
            print(f'epoch {i} training loss {totLoss:.5f} ({(totLoss-lastLoss)*100/lastLoss:+.2f}%) test loss {totTestLoss:.5f} ({(totTestLoss-lastTestLoss)*100/lastTestLoss:+.2f}%) in {span:.3f}s ({span2/span*100:.2f}% for training) lr {current_lr}')
            if testPos is not None:
                with torch.no_grad():
                    print("test eval result:", self.modelEval.get_static_eval(testPos.float().to(self.device), 0)[:, 0].tolist())
            sys.stdout.flush()
            if totTestLoss < miniLoss:
                miniLoss = totTestLoss
                self.save(fileBest, self.modelEval)
            elif totTestLoss > lastTestLoss:
                current_lr /= 10**.5
                for g in self.optimizer.param_groups:
                    g['lr'] = current_lr
            lastTestLoss = totTestLoss
            lastLoss = totLoss
        
        for g in self.optimizer.param_groups:
            g['lr'] = self.lr

    def get_int(self, Stensor, Q):
        tensor = int(round(float(Stensor)*Q))
        self.maxi = max(self.maxi, abs(tensor))
        return tensor.to_bytes(1, sys.byteorder, signed=True)

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
                    f.write(self.get_int(model.tohidden.weight[j][i], model.QA))
            for i in range(model.HLSize):
                f.write(self.get_int(model.tohidden.bias[i], model.QA))
            for i in range(model.HLSize*2):
                for j in range(model.BUCKET):
                    f.write(self.get_int(model.toout.weight[j][i], model.QB))
            for i in range(model.BUCKET):
                f.write(self.get_int(model.toout.bias[i], model.QB))
        endTime = time.time()
        print(f'save model to {filename} (max |weight| {self.maxi}) in {endTime-startTime:.3f}s')
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
                    for j in range(self.model.BUCKET):
                        self.model.toout.weight[j][i] = self.read_bytes(f.read(1))/self.model.QB
                for i in range(self.model.BUCKET):
                    self.model.toout.bias[i] = self.read_bytes(f.read(1))/self.model.QB
