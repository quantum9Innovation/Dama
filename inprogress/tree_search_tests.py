from pythonchess import chess
import numpy as np


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
            w_score += 3

        if x[i] == 6:
            b_score += 3

        if x[i] == 7:
            w_score += 3

        if x[i] == 8:
            b_score += 3

        if x[i] == 9:
            w_score += 0

        if x[i] == 10:
            b_score += 0

        if x[i] == 11:
            w_score += 9

        if x[i] == 12:
            b_score += 9

    if turn == 13:
        return 1 / (1 + np.exp((b_score - w_score) / 5))
    else:
        return 1 / (1 + np.exp((w_score - b_score) / 5))


def model_out(turn, board):

    fen = board.board_fen().split('/')
    net_input = []

    for line in range(len(fen)):

        fen_string = list(fen[line])
        for char in fen_string:

            if char == 'p':
                net_input.append(1)
            elif char == 'P':
                net_input.append(2)
            elif char == 'r':
                net_input.append(3)
            elif char == 'R':
                net_input.append(4)
            elif char == 'n':
                net_input.append(5)
            elif char == 'N':
                net_input.append(6)
            elif char == 'b':
                net_input.append(7)
            elif char == 'B':
                net_input.append(8)
            elif char == 'k':
                net_input.append(9)
            elif char == 'K':
                net_input.append(10)
            elif char == 'q':
                net_input.append(11)
            elif char == 'Q':
                net_input.append(12)
            else:
                num_squares = int(char)
                for i in range(num_squares):
                    net_input.append(0)

    if turn:
        net_input.append(13)
    else:
        net_input.append(14)

    net_input = np.array(net_input)
    score = evaluate(net_input)
    return score
    # score = model.predict(np.array([net_input]))
    # return score[0][0]


board = chess.Board()

board.push_san("e4")
board.push_san("e5")
board.push_san("Qh5")
board.push_san("Nc6")
board.push_san("Bc4")
board.push_san("Nf6")
board.push_san("Qxf7")

print(board)
print(model_out(False, board))
print(model_out(True, board))
