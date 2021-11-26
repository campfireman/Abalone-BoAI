import numpy as np
from abalone_engine.game import Game, Move
from game_static import POSSIBLE_MOVES, s_get_legal_moves


def test_s_get_valid_moves():
    TEST_BOARD = np.array([
        # 0  1  2  3  4  5  6  7  8
        [0, 0, 0, 0, -1, -1, -1, -1, -1],  # 0
        [0, 0, 0, -1, -1, -1, -1, -1, -1],  # 1
        [0, 0, 0, 0, -1, -1, -1, 0, 0],  # 2
        [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 3
        [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 4
        [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 5
        [0, 0, 1, 1, 1, 0, 0, 0, 0],  # 6
        [1, 1, 1, 1, 1, 1, 0, 0, 0],  # 7
        [1, 1, 1, 1, 1, 0, 0, 0, 0],  # 8
    ], dtype='int')
    legal_moves = s_get_legal_moves(TEST_BOARD, 1)
    for valid_move in Game.s_generate_legal_moves(TEST_BOARD, 1):
        # print(Move.from_original(valid_move).to_standard())
        index = POSSIBLE_MOVES[Move.from_original(valid_move).to_standard()]
        assert legal_moves[index] == 1
