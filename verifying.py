from tkinter import *
import sys
MAX_BIG = (1<<64)-1
file = sys.argv[1]
constants = []
table = []
blue = "#0000FF"
violet = "#FF00FF"
red = "#FF0000"
white = "#FFFFFF"
yellow = "#FFFF00"
with open(file) as f:
    L = list(map(int, f.readline().split()))
    index = 0
    while 1:
        constants.append((L[index], L[index+1], L[index+2]))
        index += 3
        table.append(L[index+1:index+1+L[index]])
        index += L[index]+1
        if(index >= len(L)):break
print(len(constants))
print(len(table))

def print_mask(mask):
    for i in range(8):
        for i in range(8):
            print(mask&1, end='')
            mask >>= 1
        print()

def get_moves(mask, case):
    #print_mask(mask)
    key = (mask*constants[case][0] & (MAX_BIG >> constants[case][1])) >> (64-constants[case][1]-constants[case][2])
    #print(key)
    return table[case][key]

clipped_row = [0]*8
clipped_col = [0]*8
def init_lines():
    row = MAX_BIG >> (8*7+2) << 1
    col = 0x0001010101010100
    for i in range(8):
        clipped_row[i] = row
        clipped_col[i] = col
        row <<= 8
        col <<= 1

place_tower = -1
types = [0]*64

def recalculate():
    global place_tower
    if place_tower == -1:
        for id, b in enumerate(buttons):
            b.config(bg=[white, blue][types[id]])
    else:
        col = place_tower & 7
        row = place_tower >> 3
        mask = pieces&((clipped_col[col]|clipped_row[row])&(MAX_BIG-(1 << place_tower)))
        moves = get_moves(mask, place_tower)
        for index in range(64):
            mask_square = 1 << index
            if(index == place_tower):
                color = yellow
            elif (moves & mask_square) and (pieces & mask_square):
                color = violet
            elif (moves & mask_square):
                color = red
            elif (pieces & mask_square):
                color = blue
            else: 
                color = white
            buttons[index].config(bg=color)

def get_command(index):
    def _command():
        global pieces, place_tower
        if place_tower != -1:
            if types[index] == 2:
                types[index] = 0
                place_tower = -1
            else:
                pieces ^= 1 << index
                types[index] ^= 1
        else:
            pieces ^= 1 << index
            if types[index] == 1:place_tower = index
            types[index] += 1
        recalculate()
    return _command        

pieces = 0
init_lines()
cellSize = 50
size = 8*50
fen = Tk()
fen.geometry(f'{size}x{size}')

buttons = [Button(fen, bg=white, command=get_command(i)) for i in range(64)]
x = 0
y = 0
for button in buttons:
    button.place(x=x*cellSize, y=y*cellSize, width=50, height=50)
    x += 1
    if(x >= 8):
        x = 0
        y += 1

fen.mainloop()
