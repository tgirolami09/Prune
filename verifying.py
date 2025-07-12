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
    if is_rook.cget('text') == 'rook':
        case += 64
    key = (mask*constants[case][0] & (MAX_BIG >> constants[case][1])) >> (64-constants[case][1]-constants[case][2])
    #print(key)
    return table[case][key]

clipped_row = [0]*8
clipped_col = [0]*8
clipped_diag = [0]*15
clipped_idiag = [0]*15
clipped_mask = (MAX_BIG >> 16 << 8) & (MAX_BIG-0x8181818181818181);
def init_lines():
    row = MAX_BIG >> (8*7+2) << 1
    col = 0x0001010101010100
    for i in range(8):
        clipped_row[i] = row
        clipped_col[i] = col
        row <<= 8
        col <<= 1
    diag = 0
    idiag = 0
    for i in range(15):
        diag <<= 8
        if(i < 8):diag |= 1 << i
        idiag <<= 8
        if(i < 8):idiag |= 1 << (7-i)
        clipped_diag[i] = diag&clipped_mask
        clipped_idiag[i] = idiag&clipped_mask

place_piece = -1
types = [0]*64

def propagate(mask, square, directions):
    for dir in directions:
        counter = 0
        col, row = (square&7)+dir[0], (square >> 3)+dir[1]
        while 0 <= col < 8 and 0 <= row < 8 and mask&(1 << ((row << 3) | col)):
            counter += 1
            col += dir[0]
            row += dir[1]
        yield counter

def nb_poss(mask, square):
    res = 1
    directions = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
    for d, m in zip(directions, propagate(mask, square, directions)):
        res *= max(m, 1)
    return res

def recalculate():
    global place_piece
    if place_piece == -1:
        for id, b in enumerate(buttons):
            b.config(bg=[white, blue][types[id]])
    else:
        col = place_piece & 7
        row = place_piece >> 3
        if is_rook.cget('text') == 'rook':
            mask_pos_moves = clipped_col[col]|clipped_row[row]
        else:
            mask_pos_moves = clipped_diag[col+row]|clipped_idiag[row-col+7]
        mask = pieces&(mask_pos_moves&(MAX_BIG-(1 << place_piece)))
        fen.title(f"{place_piece} : {nb_poss(get_moves(0, place_piece), place_piece)}/{len(table[place_piece+64 if is_rook.cget('text') == 'rook' else place_piece])}")
        moves = get_moves(mask, place_piece)
        for index in range(64):
            mask_square = 1 << index
            if(index == place_piece):
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
        global pieces, place_piece
        print(f"Index is {index}")
        if place_piece != -1:
            if types[index] == 2:
                types[index] = 0
                place_piece = -1
            else:
                pieces ^= 1 << index
                types[index] ^= 1
        else:
            pieces ^= 1 << index
            if types[index] == 1:place_piece = index
            types[index] += 1
        recalculate()
    return _command        

def change_rook():
    if is_rook.cget('text') == 'rook':
        is_rook.config(text='bishop')
    else:
        is_rook.config(text='rook')
    recalculate()

pieces = 0
init_lines()
cellSize = 50
size = 8*50
fen = Tk()
fen.geometry(f'{size}x{size+cellSize}')

def make_board_label(i):
    lbl = Label(fen, bg=white, borderwidth=1, relief="solid")
    lbl.bind("<Button-1>", lambda e: get_command(i)())
    return lbl

def make_type_change_label():
    lbl = Label(fen, bg=white, borderwidth=1, relief="solid")
    lbl.bind("<Button-1>", lambda e: change_rook())
    return lbl

if sys.platform=="darwin":
    buttons = [make_board_label(i) for i in range(64)]
    is_rook = make_type_change_label()
else:
    buttons = [Button(fen, bg=white, command=get_command(i)) for i in range(64)]
    is_rook = Button(fen, bg=white, command=change_rook)
x = 0
y = 0
for button in buttons:
    button.place(x=x*cellSize, y=y*cellSize, width=cellSize, height=cellSize)
    x += 1
    if(x >= 8):
        x = 0
        y += 1

is_rook.place(x=7*cellSize//2, y=y*cellSize, width=cellSize, height=cellSize)
is_rook.config(text='rook')
fen.mainloop()
