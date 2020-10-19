from pythonchess import chess
import numpy as np
from copy import deepcopy


def formatter(board):

    raw_fen = board.fen().split(' ')
    fen = raw_fen[0].split('/')

    turn = raw_fen[1]
    castling_info = list(raw_fen[2])

    castle_k = 13
    castle_K = 14
    castle_q = 15
    castle_Q = 16

    for char in castling_info:
        if char == 'k':
            castle_k = 17
        if char == 'K':
            castle_K = 18
        if char == 'q':
            castle_q = 19
        if char == 'Q':
            castle_Q = 20

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

            if turn == 'w':

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

            else:

                if fen_string[char] == 'p':
                    net_input.append(1)
                    wpco.append([char + line_bonus, line])
                elif fen_string[char] == 'P':
                    net_input.append(2)
                    bpco.append([char + line_bonus, line])
                elif fen_string[char] == 'r':
                    net_input.append(3)
                    wrco.append([char + line_bonus, line])
                elif fen_string[char] == 'R':
                    net_input.append(4)
                    brco.append([char + line_bonus, line])
                elif fen_string[char] == 'n':
                    net_input.append(5)
                    wnco.append([char + line_bonus, line])
                elif fen_string[char] == 'N':
                    net_input.append(6)
                    bnco.append([char + line_bonus, line])
                elif fen_string[char] == 'b':
                    net_input.append(7)
                    wbco.append([char + line_bonus, line])
                elif fen_string[char] == 'B':
                    net_input.append(8)
                    bbco.append([char + line_bonus, line])
                elif fen_string[char] == 'k':
                    net_input.append(9)
                    wkco.append([char + line_bonus, line])
                elif fen_string[char] == 'K':
                    net_input.append(10)
                    bkco.append([char + line_bonus, line])
                elif fen_string[char] == 'q':
                    net_input.append(11)
                    wqco.append([char + line_bonus, line])
                elif fen_string[char] == 'Q':
                    net_input.append(12)
                    bqco.append([char + line_bonus, line])
                else:
                    num_squares = int(fen_string[char])
                    line_bonus += num_squares - 1
                    for i in range(num_squares):
                        net_input.append(0)

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
    if turn == 'b':
        piece_map = [bpco, wpco, brco, wrco, bnco, wnco, bbco, wbco, bkco, wkco, bqco, wqco]

    castling_rights = [castle_K, castle_k, castle_Q, castle_q]
    if turn == 'b':
        castling_rights = [castle_k, castle_K, castle_q, castle_Q]

    net_input.append(castling_rights[0])
    net_input.append(castling_rights[1])
    net_input.append(castling_rights[2])
    net_input.append(castling_rights[3])

    net_input = np.array(net_input)

    return [net_input, piece_map, piece_mobility, piece_threats]


board = chess.Board(fen='r2qk2r/pppb1ppp/2nbpn2/3pN3/3P1P2/1P1BPR2/PBPN2PP/R2Q2K1 b kq - 0 1')
print(board)
print(formatter(board))
