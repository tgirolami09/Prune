from tkinter import *
from math import log, sqrt, pi as PI
import sys
width = 200
height = 50
epsilon = 5
fen = Tk()
fen.title('progress')
white = "#FFFFFF"
file = sys.argv[1]
results = {}
projects = []
with open(file, 'r') as f:
    inResults = True
    for line in f:
        if line.startswith("project"):
            break
        version1, version2, result = line.split()
        result = tuple(map(int, result.split('/')))
        if version1 not in results:
            results[version1] = {}
        if version2 not in results:
            results[version2] = {}
        results[version1][version2] = result[2], result[1], result[0]
        results[version2][version1] = result

    for line in f:
        projects.append(line.replace('\n', '').split(' : '))

def force(x):
    return x/(1-x)

def diff(p):
    return -400 * log(1 / p - 1, 10)

def inverseErrorFunction(x):
    a = 8 * (PI - 3) / (3 * PI * (4 - PI))
    y = log(1 - x * x)
    z = 2 / (PI * a) + y / 2
    ret = sqrt(sqrt(z * z - y / a) - z)
    if (x < 0):
        return -ret
    return ret

def phiInv(p):
    return sqrt(2)*inverseErrorFunction(2*p-1)

def showResult(result, xStart, yStart):
    wins = result[0]
    draws = result[1]
    losses = result[2]
    total = sum(result)

    winP = wins/total
    drawP = draws/total
    lossesP = losses/total

    score = wins+draws/2
    percentage = score/total
    winsDev = winP*(1-percentage)**2
    drawsDev = drawP*(0.5-percentage)**2
    lossesDev = winP*(0-percentage)**2
    dev = sqrt(winsDev + drawsDev + lossesDev) / sqrt(total)

    confidenceP = 0.95
    minConfidenceP = (1 - confidenceP) / 2
    maxConfidenceP = 1 - minConfidenceP
    stdDeviation = sqrt(winsDev + drawsDev + lossesDev) / sqrt(total)
    devMin = percentage + phiInv(minConfidenceP) * stdDeviation
    devMax = percentage + phiInv(maxConfidenceP) * stdDeviation
    difference = (diff(devMax) - diff(devMin))/2

    eloDelta = diff(percentage)

    eloDelta = round(eloDelta, 1)
    difference = round(difference, 1)

    text = Label(canvas, text=f'{eloDelta} ± {difference}', bg=white)
    text.place(x=xStart+epsilon, y=yStart+epsilon)
    resultTexts.append(text)


order = list(results.keys())
nbResults = len(results)
nbProjects = len(projects)
totWidth = width*(nbResults+1)
totHeight=height*(nbResults+1)+height//2*nbProjects
fen.geometry(f'{totWidth}x{totHeight}')
canvas = Canvas(fen, width=totWidth, height=totHeight, bg="#FFFFFF")
canvas.place(x=0, y=0)
Heads = []
resultTexts = []
for row in range(nbResults+1):
    for col in range(nbResults+1):
        if row == 0:
            if col == 0:continue
            Heads.append(Label(canvas, text=order[col-1], bg=white))
            Heads[-1].place(x=col*width+epsilon, y=row*height+epsilon)
        elif col == 0:
            Heads.append(Label(canvas, text=order[row-1], bg=white))
            Heads[-1].place(x=col*width+epsilon, y=row*height+epsilon)
        else:
            if order[row-1] in results[order[col-1]]:
                showResult(results[order[col-1]][order[row-1]], col*width, row*height)
lines = []
endHeight = height*(nbResults+1)
for case in range(1, nbResults+2):
    lines.append(canvas.create_line(case*width, 0, case*width, endHeight))
    lines.append(canvas.create_line(0, height*case, width*(nbResults+1), height*case))
projectsLabel = []
for idProj, project in enumerate(projects):
    projectsLabel.append(Label(canvas, text=project[0], bg=white))
    projectsLabel[-1].place(x=0, y=endHeight+idProj*height//2+epsilon)

fen.mainloop()