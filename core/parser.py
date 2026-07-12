from math import *
import matplotlib.pyplot as plt
import sys
plt.ion()


def CalculateInverseErrorFunction(x):
    a = 8 * (pi - 3) / (3 * pi * (4 - pi))
    y = log(1 - x * x)
    z = 2 / (pi * a) + y / 2

    ret = sqrt( sqrt(z * z - y / a) - z)
    if (x < 0):
        return -ret
    return ret


def phiInv(p):
    return sqrt(2) * CalculateInverseErrorFunction(2 * p - 1);

def CalculateEloDifference(p):
    return 400*log(p/(1-p))

def getScore(penta):
    penta = tuple(map(int, penta[1:-2].split(', ')))
    nbGames = sum(penta)
    score = (penta[1]*0.25+penta[2]*0.5+penta[3]*0.75+penta[4])/nbGames
    elo = 400*log(score/(1-score))

    pentaP = tuple(i/nbGames for i in penta)
    stdDeviation = sqrt(sum(
        pentaP[i]*(i/5-score)**2
        for i in range(5)
    ))/sqrt(nbGames)
    confidenceP = 0.95
    minConfidenceP = (1 - confidenceP) / 2
    maxConfidenceP = 1 - minConfidenceP
    devMin = score + phiInv(minConfidenceP) * stdDeviation
    devMax = score + phiInv(maxConfidenceP) * stdDeviation
    difference = CalculateEloDifference(devMax) - CalculateEloDifference(devMin)
    return elo, difference / 2
Xs = []
Ys = []
Yerr = []
for line in sys.stdin:
    Xs.append(int(line.split()[0]))
    elo, error = getScore(line.split(': ')[1])
    Ys.append(elo)
    Yerr.append(error)
plt.errorbar(Xs, Ys, Yerr, marker='s')
plt.show(block=True)
while 1:pass