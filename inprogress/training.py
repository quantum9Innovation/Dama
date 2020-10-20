"""
THIS IS A MODIFIED VERSION OF THE `training.py` FILE.
THIS VERSION INCLUDES 4 DIFFERENT VIEWS OF THE CHESS BOARD WHICH ARE FED INTO THE NN.
STATUS: "IN DEVELOPMENT"
"""

import copy
import tensorflow as tf
import keras as kr
import numpy as np
from pythonchess import chess
from copy import deepcopy

# TODO: Get the dataset ready
# TODO: Write basic training algorithm
# TODO: Advanced training & next steps ...

# INPUTS

board_in = kr.Input(shape=(68,))  # board-centric view; omniscient
piece_in = kr.Input(shape=(64,))  # piece-centric view; omniscient
mobility_in = kr.Input(shape=(64,))  # square-centric view; piece mobility
pressure_in = kr.Input(shape=(64,))  # square-centric view; square pressure

# PREPROCESSING

"""
VOCABULARY SIZE = 2 players * (6 pieces + 4 castling possibilities) + blank squares
                = 2 * 10 + 1
                = 21 (final vocabulary size)
"""

board = kr.layers.Embedding(21, 4, input_length=68)(board_in)
board_1 = kr.layers.Flatten()(board)
"""
The `piece` array should already be preprocessed to avoid any complications with 
nonexistent pieces
"""

# SUB-NETWORKS

board_2 = kr.layers.Dense(64, activation='softplus')(board_1)
board_3 = kr.layers.Dense(8, activation='softplus')(board_2)

piece = kr.layers.Dense(8, activation='softplus')(piece_in)
mobility = kr.layers.Dense(8, activation='softplus')(mobility_in)
pressure = kr.layers.Dense(8, activation='softplus')(pressure_in)

# CONCATENATION

out = kr.layers.Concatenate()([board_3, piece, mobility, pressure])
out_1 = kr.layers.Dense(32, activation='softplus')(out)
out_2 = kr.layers.Dense(8, activation='softplus')(out_1)
out_3 = kr.layers.Dense(3, activation='softmax')(out_2)

# KERAS MODEL

"""
Output sequence should be as follows:
    NEURON 1: Side with last move win probability
    NEURON 2: Draw probability
    NEURON 3: Side with last move lose probability --(1)
    
    (1) Return this value to the `minimax` algorithm
        Once a move is played on the board the nn will give p_win to the side whose turn it is to play
        Instead return that side's loss probability = desired minimax win probability
"""
Model = kr.Model(inputs=[board_in, piece_in, mobility_in, pressure_in], outputs=out_3, name='dama')
opt_adadelta = kr.optimizers.Adadelta(rho=0.9)
Model.compile(loss='categorical_crossentropy', optimizer=opt_adadelta)
Model.summary()


def model_eval(data):
    """Returns losing probability (see #KERAS_MODEL for more information"""

    inp1 = np.array(data[0])

    # flatten input #2 (pieces list)
    inp2 = []
    for stack in data[1]:
        for el in stack:
            inp2.append(el[0])
            inp2.append(el[1])

    inp2 = np.array(inp2)

    inp3 = np.array(data[2])
    inp4 = np.array(data[3])

    inp1 = inp1.reshape((1, 68))
    inp2 = inp2.reshape((1, 64))
    inp3 = inp3.reshape((1, 64))
    inp4 = inp4.reshape((1, 64))

    return float(Model([inp1, inp2, inp3, inp4])[0][2])


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
        wpco.append([16, 16])
    while len(bpco) < 8:
        bpco.append([16, 16])
    while len(wrco) < 2:
        wpco.append([16, 16])
    while len(brco) < 2:
        brco.append([16, 16])
    while len(wnco) < 2:
        wpco.append([16, 16])
    while len(bnco) < 2:
        bnco.append([16, 16])
    while len(wbco) < 2:
        wpco.append([16, 16])
    while len(bbco) < 2:
        bbco.append([16, 16])

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

    return model_eval([net_input, piece_map, piece_mobility, piece_threats])


print('\nStarting Game Analysis: ')
print(formatter(chess.Board()))

board = chess.Board()
rounds = [5, 5]


def minimax_tree_search(board, depth):

    if board.is_game_over():

        if board.is_checkmate():

            if board.turn:
                return board, 0
            else:
                return board, 1

        elif board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or \
        board.is_fivefold_repetition() or board.can_claim_draw():
            return board, 0

        elif board.has_insufficient_material(board.turn):
            return board, 0

    if depth == 0:
        turn = board.turn
        if turn == chess.WHITE:
            turn = True
        else:
            turn = False
        return board, formatter(board)

    scores = []

    new_boards = []
    for move in board.generate_legal_moves():

        new_boards.append(copy.deepcopy(board))
        new_boards[-1].push_uci(str(move))
        turn = new_boards[-1].turn
        if turn == chess.WHITE:
            turn = True
        else:
            turn = False
        scores.append((new_boards[-1], formatter(new_boards[-1])))

    sum = 0
    for el in scores:
        sum += el[1]

    best_scores = []
    while len(best_scores) < min(rounds[len(rounds) - depth - 1], len(scores)):

        random_sample = int(np.round(np.random.random() * len(scores) - 1))
        choice_probability = np.random.random()

        if choice_probability <= scores[random_sample][1] / sum:
            best_scores.append(scores[random_sample])

    """for best one-time performance (non-generative)
    best_scores = [(None, 0)] * min(rounds[len(rounds) - depth - 1], len(scores))
    for score in scores:
        for i in range(len(best_scores)):
            if score[1] > best_scores[i][1]:
                best_scores.insert(i, score)
                del best_scores[-1]
    """

    best_score = 0
    best_move = ''
    for considered_move in best_scores:

        sub_score = 1 - minimax_tree_search(considered_move[0], depth - 1)[1]

        if sub_score > best_score:
            best_score = sub_score
            best_move = considered_move[0]

    return best_move, best_score


def start_self_play():

    game_history = [chess.Board()]
    game_over = False

    print(game_history[-1])

    while not game_over:

        current_board = copy.deepcopy(game_history[-1])
        game_history.append(minimax_tree_search(current_board, len(rounds))[0])

        if game_history[-1].is_game_over():
            game_over = True

        print('\n\n')
        print('Move #%s' % (len(game_history) - 1))
        print(game_history[-1])


# TODO: Fix unnecessary memory usage and crashing (usually ~move #5-7)
start_self_play()
