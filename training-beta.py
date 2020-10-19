"""
THIS IS A MODIFIED VERSION OF THE `training.py` FILE.
THIS VERSION INCLUDES 4 DIFFERENT VIEWS OF THE CHESS BOARD WHICH ARE FED INTO THE NN.
STATUS: "IN DEVELOPMENT"
"""

import copy
import keras as kr
import numpy as np
from pythonchess import chess
from copy import deepcopy


# INPUTS

board_in = kr.Input(shape=(68,))  # board-centric view; omniscient
piece_in = kr.Input(shape=(64,))  # piece-centric view; omniscient
mobility_in = kr.Input(shape=(64,))  # square-centric view; piece mobility
pressure_in = kr.Input(shape=(64,))  # square-centric view; square pressure

# PREPROCESSING

board = kr.layers.Embedding(16, 4, input_length=68)(board_in)
board = kr.layers.Flatten()(board)
"""
The `piece` array should already be preprocessed to avoid any complications with 
nonexistent pieces
"""

# SUB-NETWORKS

board = kr.layers.Dense(64, activation='softplus')(board)
board = kr.layers.Dense(8, activation='softplus')(board)

piece = kr.layers.Dense(8, activation='softplus')(piece_in)
mobility = kr.layers.Dense(8, activation='softplus')(mobility_in)
pressure = kr.layers.Dense(8, activation='softplus')(pressure_in)

# CONCATENATION

out = kr.layers.Concatenate()([board, piece, mobility, pressure])
out = kr.layers.Dense(32, activation='softplus')(out)
out = kr.layers.Dense(8, activation='softplus')(out)
out = kr.layers.Dense(3, activation='softmax')(out)

# KERAS MODEL

"""
Output sequence should be as follows:
    NEURON 1: Side with last move win probability
    NEURON 2: Draw probability
    NEURON 3: Side with last move lose probability
"""
Model = kr.Model(inputs=[board_in, piece_in, mobility_in, pressure_in], outputs=out, name='dama')
opt_adadelta = kr.optimizers.Adadelta(rho=0.9)
Model.compile(loss='categorical_crossentropy', optimizer=opt_adadelta)
Model.summary()


# TODO: Format input `x` correctly before giving as input to nn
def evaluate(x):

    turn = x[-1]
    w_score = 0
    b_score = 0

    for i in range(0, len(x) - 1):

        if x[i] == 1:
            w_score+=1

        if x[i] == 2:
            b_score+=1

        if x[i] == 3:
            w_score+=5

        if x[i] == 4:
            b_score+=5

        if x[i] == 5:
            w_score += 2.5

        if x[i] == 6:
            b_score += 2.5

        if x[i] == 7:
            w_score += 3.5

        if x[i] == 8:
            b_score += 3.5

        if x[i] == 9:
            w_score += 0

        if x[i] == 10:
            b_score += 0

        if x[i] == 11:
            w_score += 9

        if x[i] == 12:
            b_score += 9

    if turn == 13:
        return 1 / (1 + np.exp((b_score - w_score) * 0.4))
    else:
        return 1 / (1 + np.exp((w_score - b_score) * 0.4))


# TODO: Use `evaluate(board)` to get model predictions
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


"""
REMOVED UNTIL NEURAL NETWORK IS FULLY CONFIGURED
"""
# print('\nStarting Game Analysis: ')
# print(model_out(False, chess.Board()))

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
        return board, model_out(turn, board)

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
        scores.append((new_boards[-1], model_out(turn, new_boards[-1])))

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


"""
REMOVED UNTIL NEURAL NETWORK IS FULLY CONFIGURED
"""
# start_self_play()
