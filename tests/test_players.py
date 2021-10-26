from ctypes import c_char_p
from multiprocessing import Manager, Process
from typing import List, Optional

from abalone_engine import players
from abalone_engine.enums import (Direction, InitialPosition, Marble, Player,
                                  Space)
from abalone_engine.game import Game, Move

BOARD_DIMENSIONS = [
    5, 6, 7, 8, 9, 8, 7, 6, 5
]


def _validate_board(self, board: List[List]):
    assert len(board) == len(BOARD_DIMENSIONS)
    for i, row in enumerate(board):
        assert len(row) == BOARD_DIMENSIONS[i]


def test_heuristic():
    boards = [
        {
            'value': InitialPosition.DEFAULT.value,
            'in_turn': Player.BLACK,
            'expected_adjacency_black': 54,
            'expected_adjacency_white': 54,
            'expected_distance_black': 46,
            'expected_distance_white': 46,
        },
        {
            'value': InitialPosition.GERMAN_DAISY.value,
            'in_turn': Player.BLACK,
            'expected_adjacency_black': 48,
            'expected_adjacency_white': 48,
            'expected_distance_black': 42,
            'expected_distance_white': 42,
        },
        {
            'value': [
                [Marble.BLANK] * 5,
                [Marble.BLANK] * 6,
                [Marble.BLANK] * 7,
                [Marble.BLANK] * 3 + [Marble.BLACK] * 2 + [Marble.BLANK] * 3,
                [Marble.BLANK] * 3 + [Marble.BLACK] * 3 + [Marble.BLANK] * 3,
                [Marble.BLANK] * 3 + [Marble.BLACK] * 2 + [Marble.BLANK] * 3,
                [Marble.BLANK] * 7,
                [Marble.BLANK] * 6,
                [Marble.BLANK] * 5,
            ],
            'in_turn': Player.BLACK,
            'expected_adjacency_black': 24,
            'expected_adjacency_white': 0,
            'expected_distance_black': 6,
            'expected_distance_white': 0,
        },
        {
            'value': [
                [Marble.BLANK] * 5,
                [Marble.BLANK] * 6,
                [Marble.BLANK] * 7,
                [Marble.BLANK] * 8,
                [Marble.BLANK] * 9,
                [Marble.BLANK] * 3 + [Marble.BLACK] * 2 + [Marble.BLANK] * 3,
                [Marble.BLANK] * 2 + [Marble.BLACK] * 3 + [Marble.BLANK] * 2,
                [Marble.BLANK] * 2 + [Marble.BLACK] * 2 + [Marble.BLANK] * 2,
                [Marble.BLANK] * 5,
            ],
            'in_turn': Player.BLACK,
            'expected_adjacency_black': 24,
            'expected_adjacency_white': 0,
            'expected_distance_black': 14,
            'expected_distance_white': 0,
        },
    ]

    game = Game()

    for board in boards:
        game.board = board['value']
        game.turn = board['in_turn']
        game.marbles = game.init_marbles()
        algorithm = players.AlphaBetaSimple(game, game.turn.value)
        counts = algorithm._count_heuristics(game)
        print(game)
        print(algorithm._heuristic(game))
        assert counts['sum_adjacency'][Player.BLACK.value] == board['expected_adjacency_black']
        assert counts['sum_adjacency'][Player.WHITE.value] == board['expected_adjacency_white']
        assert counts['sum_distance'][Player.BLACK.value] == board['expected_distance_black']
        assert counts['sum_distance'][Player.WHITE.value] == board['expected_distance_white']


class TestAbaProPlayer:
    def setup_method(self):
        self.aba_pro = players.AbaProPlayer()

    def test_pipe(self):
        def runner(func, result):
            process_result = func()
            ret_value.value = process_result

        message = 'test'
        manager = Manager()
        ret_value = manager.Value(c_char_p, "")
        proc = Process(target=runner, args=(self.aba_pro.read_move, ret_value))
        proc.start()

        self.aba_pro.send_move(
            message, pipe_path=players.AbaProPlayer.RECIEVING_PIPE)
        proc.join()

        assert ret_value.value == message
