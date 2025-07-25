import sys
import subprocess
from tqdm import tqdm
def pushCommand(prog, command):
    prog.stdin.write(command.encode())
    prog.stdin.flush()

def readResult(prog):
    dataMoves = {}
    markEnd = 'Nodes searched: '
    while 1:
        line = prog.stdout.readline().decode('utf-8')
        line = line.replace('\n', '')
        if line.startswith(markEnd):
            break
        else:
            L = line.split(": ")
            if len(L) == 2 and (len(L[0]) == 4 or len(L[0]) == 5) and L[1].isdigit():
                dataMoves[L[0]] = L[1]
    return dataMoves, int(line[len(markEnd):])

file = sys.argv[1]
with open(file) as f:
    listPos = f.readlines()
prog = subprocess.Popen([sys.argv[2]], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
tq1 = tqdm(listPos)
for pos in tq1:
    parts = pos.split(';')
    fen = parts[0]
    tq2 = tqdm(parts[1:], leave=False)
    for tests in tq2:
        depth, res = tests.split()
        depth = int(depth[1:])
        pushCommand(prog, f"position fen {fen}\n")
        pushCommand(prog, f"go perft {depth}\n")
        detailed, sumup = readResult(prog)
        if int(sumup) != int(res):
            tq2.close()
            tq1.write(f'fen={fen} depth={depth} expected={res} res={sumup}')
            break

