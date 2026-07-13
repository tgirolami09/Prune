from math import *
import matplotlib.pyplot as plt
import sys
plt.style.use('dark_background')
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

def CalculateEloDifference(p, variance):
    return (p-0.5)/sqrt(2*variance)*(800/log(10))

def getScore(penta):
    penta = tuple(map(int, penta[1:-2].split(', ')))
    nbGames = sum(penta)
    score = (penta[1]*0.25+penta[2]*0.5+penta[3]*0.75+penta[4])/nbGames

    pentaP = tuple(i/nbGames for i in penta)
    variance = sum(
        pentaP[i]*(i/4-score)**2
        for i in range(5)
    )
    elo = CalculateEloDifference(score, variance)
    stdDeviation = sqrt(variance/nbGames)
    confidenceP = 0.95
    minConfidenceP = (1 - confidenceP) / 2
    maxConfidenceP = 1 - minConfidenceP
    devMin = score + phiInv(minConfidenceP) * stdDeviation
    devMax = score + phiInv(maxConfidenceP) * stdDeviation
    difference = CalculateEloDifference(devMax, variance) - CalculateEloDifference(devMin, variance)
    print(penta, elo, difference/2)
    return elo, difference / 2
fig, ax = plt.subplots()
Xs = [[]]
Ys = [[]]
Yerr = [[]]
for line in sys.stdin:
    idS = int(line.split()[0])
    if(Xs[-1] and idS < Xs[-1][-1]):
        Xs.append([])
        Ys.append([])
        Yerr.append([])
    Xs[-1].append(idS)
    elo, error = getScore(line.split(': ')[1])
    Ys[-1].append(elo)
    Yerr[-1].append(error)
for i, X, Y, Ye in zip(range(len(Xs)), Xs, Ys, Yerr):
    ax.errorbar([i/X[-1] for i in X], Y, Ye, marker='s', label=str(2**i)+'ppp', capsize=5)
ax.legend()
plt.show(block=True)