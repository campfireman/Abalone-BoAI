import time

import numpy as np

from abalone_engine.game import Game
from game_static import s_get_legal_moves


def test_perf(func, label, n=300):
    times = []
    for _ in range(0, n):
        start = time.time()
        func()
        end = time.time()
        times.append(end - start)
    print(f'{label}: {np.average(times)}s')


def main():
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

    def run_old():
        list(Game.s_generate_legal_moves(TEST_BOARD, 1))

    def run_new():
        s_get_legal_moves(TEST_BOARD, 1)

    test_perf(run_old, 'Old')
    test_perf(run_new, 'New')


if __name__ == '__main__':
    main()
