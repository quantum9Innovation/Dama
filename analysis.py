from pythonchess import chess
import numpy as np
from copy import deepcopy


def model_out(board):

    raw_fen = board.fen().split(' ')
    fen = raw_fen[0].split('/')

    turn = raw_fen[1]
    castling_info = list(raw_fen[2])

    castle_k = False
    castle_K = False
    castle_q = False
    castle_Q = False

    for char in castling_info:
        if char == 'k':
            castle_k = True
        if char == 'K':
            castle_K = True
        if char == 'q':
            castle_q = True
        if char == 'Q':
            castle_Q = True

    net_input = []
    piece_mobility = [0] * 64
    piece_threats = [0] * 64

    analysis_board = deepcopy(board)

    analysis_board.turn = chess.WHITE
    for move in analysis_board.legal_moves:

        to_index = move.to_square
        from_index = move.from_square
        piece_mobility[from_index] += 1
        piece_threats[to_index] += 1

    analysis_board.turn = chess.BLACK
    for move in analysis_board.legal_moves:

        to_index = move.to_square
        from_index = move.from_square
        piece_mobility[from_index] += 1
        piece_threats[to_index] += 1

    piece_mobility = np.array(piece_mobility)
    piece_threats = np.array(piece_threats)

    piece_map = []
    wpco = []
    bpco = []
    wrco = []
    brco = []
    wnco = []
    bnco = []
    wbco = []
    bbco = []
    wkco = []
    bkco = []
    wqco = []
    bqco = []

    for line in range(len(fen)):

        fen_string = list(fen[line])
        line_bonus = 0
        for char in range(0, len(fen_string)):

            if fen_string[char] == 'P':
                net_input.append(1)
                wpco.append([char + line_bonus, line])
            elif fen_string[char] == 'p':
                net_input.append(2)
                bpco.append([char + line_bonus, line])
            elif fen_string[char] == 'R':
                net_input.append(3)
                wrco.append([char + line_bonus, line])
            elif fen_string[char] == 'r':
                net_input.append(4)
                brco.append([char + line_bonus, line])
            elif fen_string[char] == 'N':
                net_input.append(5)
                wnco.append([char + line_bonus, line])
            elif fen_string[char] == 'n':
                net_input.append(6)
                bnco.append([char + line_bonus, line])
            elif fen_string[char] == 'B':
                net_input.append(7)
                wbco.append([char + line_bonus, line])
            elif fen_string[char] == 'b':
                net_input.append(8)
                bbco.append([char + line_bonus, line])
            elif fen_string[char] == 'K':
                net_input.append(9)
                wkco.append([char + line_bonus, line])
            elif fen_string[char] == 'k':
                net_input.append(10)
                bkco.append([char + line_bonus, line])
            elif fen_string[char] == 'Q':
                net_input.append(11)
                wqco.append([char + line_bonus, line])
            elif fen_string[char] == 'q':
                net_input.append(12)
                bqco.append([char + line_bonus, line])
            else:
                num_squares = int(fen_string[char])
                line_bonus += num_squares - 1
                for i in range(num_squares):
                    net_input.append(0)

    net_input = np.array(net_input)

    while len(wpco) < 8:
        wpco.append([])
    while len(bpco) < 8:
        bpco.append([])
    while len(wrco) < 2:
        wpco.append([])
    while len(brco) < 2:
        brco.append([])
    while len(wnco) < 2:
        wpco.append([])
    while len(bnco) < 2:
        bnco.append([])
    while len(wbco) < 2:
        wpco.append([])
    while len(bbco) < 2:
        bbco.append([])

    piece_map = [wpco, bpco, wrco, brco, wnco, bnco, wbco, bbco, wkco, bkco, wqco, bqco]

    return [turn, [castle_K, castle_k, castle_Q, castle_q], net_input.reshape((8, 8)), piece_map,
            piece_mobility.reshape((8, 8)), piece_threats.reshape((8, 8))]


board = chess.Board(fen='r2q1rk1/pb3ppp/1pnbpn2/2pp4/2PP4/P1N1PNP1/1P1B1PBP/R2QK2R w KQ - 0 1')
print(board)
print(model_out(board))
