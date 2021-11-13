# -*- coding: utf-8 -*-

# Copyright 2020 Scriptim (https://github.com/Scriptim)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Unit tests for `abalone_engine.game`"""

import time
import unittest
from typing import List, Tuple, Union

import numpy as np
from abalone_engine.enums import Direction, Marble, Player, Space
from abalone_engine.game import Game, IllegalMoveException, Move


class TestGame(unittest.TestCase):
    """Test case for `abalone_engine.game.Game`."""

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
    ], dtype=float)

    def test_from_array(self):
        test_player = Player.BLACK
        result_game = Game.from_array(self.TEST_BOARD, test_player.value)
        np.testing.assert_array_almost_equal(
            result_game.to_array(), self.TEST_BOARD)
        result_game.move(Space.A1, Direction.NE)
        TEST_BOARD = np.array([[0,  0,  0,  0, - 1, - 1, - 1, - 1, - 1, ],
                              [0,  0,  0, - 1, - 1, - 1, - 1, - 1, - 1, ],
                              [0,  0,  0,  0, - 1, - 1, - 1,  0,  0, ],
                              [0,  0,  0,  0,  0,  0,  0,  0,  0, ],
                              [0,  0,  0,  0,  0,  0,  0,  0,  0, ],
                              [0,  0,  0,  1,  0,  0,  0,  0,  0, ],
                              [0,  0,  1,  1,  1,  0,  0,  0,  0, ],
                              [1,  1,  1,  1,  1,  1,  0,  0,  0, ],
                              [0,  1,  1,  1,  1,  0,  0,  0,  0, ]], dtype=float)
        np.testing.assert_array_almost_equal(
            result_game.to_array(), TEST_BOARD)
        result_game = Game.from_array(TEST_BOARD, test_player.value)
        np.testing.assert_array_almost_equal(
            result_game.to_array(), TEST_BOARD)
        self.assertEqual(result_game.turn, test_player)

    def test_switch_player(self):
        """Test `abalone_engine.game.Game.switch_player`"""
        game = Game()
        game.switch_player()
        self.assertIs(game.turn, Player.WHITE)
        game.switch_player()
        self.assertIs(game.turn, Player.BLACK)

    def test_get_marble(self):
        """Test `abalone_engine.game.Game.get_marble`"""
        game = Game()
        self.assertIs(game.get_marble(Space.A1), Marble.BLACK)
        self.assertIs(game.get_marble(Space.E1), Marble.BLANK)
        self.assertIs(game.get_marble(Space.I5), Marble.WHITE)
        self.assertRaises(Exception, lambda: game.get_marble(Space.OFF))

    def test_set_marble(self):
        """Test `abalone_engine.game.Game.set_marble`"""
        game = Game()
        game.set_marble(Space.A1, Marble.BLACK)
        self.assertIs(game.get_marble(Space.A1), Marble.BLACK)
        game.set_marble(Space.A1, Marble.WHITE)
        self.assertIs(game.get_marble(Space.A1), Marble.WHITE)
        game.set_marble(Space.A1, Marble.BLANK)
        self.assertIs(game.get_marble(Space.A1), Marble.BLANK)
        self.assertRaises(Exception, lambda: game.set_marble(
            Space.OFF, Marble.BLANK))

    def test_canonical_board(self):
        test_board = np.array([
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
        ], dtype=float)
        game = Game()
        np.testing.assert_array_almost_equal(
            game.canonical_board(), test_board)
        inverted_test_board = np.array([
            # 0  1  2  3  4  5  6  7  8
            [0, 0, 0, 0, 1, 1, 1, 1, 1],  # 0
            [0, 0, 0, 1, 1, 1, 1, 1, 1],  # 1
            [0, 0, 0, 0, 1, 1, 1, 0, 0],  # 2
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 3
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 4
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 5
            [0, 0, -1, -1, -1, 0, 0, 0, 0],  # 6
            [-1, -1, -1, -1, -1, -1, 0, 0, 0],  # 7
            [-1, -1, -1, -1, -1, 0, 0, 0, 0],  # 8
        ], dtype=float)
        game = Game(first_turn=Player.WHITE)
        np.testing.assert_array_almost_equal(
            game.canonical_board(), inverted_test_board)

    def test_get_score(self):
        """Test `abalone_engine.game.Game.get_score`"""
        game = Game()
        self.assertTupleEqual(game.get_score(), (14, 14))
        game.set_marble(Space.A1, Marble.BLANK)
        self.assertTupleEqual(game.get_score(), (13, 14))

    def test_move(self):
        """Test `abalone_engine.game.Game.move` including `abalone_engine.game.Game.move_inline` and\
        `abalone_engine.game.Game.move_broadside`"""

        game = Game()

        def assert_states(states: List[Tuple[Space, Marble]]) -> None:
            for space, marble in states:
                self.assertIs(game.get_marble(space), marble)

        # inline
        game.move(Space.B1, Direction.NORTH_EAST)
        assert_states([(Space.B1, Marble.BLANK), (Space.C2, Marble.BLACK)])
        game.move(Space.B2, Direction.NORTH_WEST)
        assert_states([(Space.D2, Marble.BLACK), (Space.C2,
                                                  Marble.BLACK), (Space.B2, Marble.BLANK)])
        game.move(Space.A2, Direction.NORTH_EAST)
        assert_states([(Space.D5, Marble.BLACK), (Space.C4, Marble.BLACK), (Space.B3, Marble.BLACK),
                       (Space.A2, Marble.BLANK)])
        # "Only own marbles may be moved"
        self.assertRaises(IllegalMoveException, lambda: game.move(
            Space.G5, Direction.SOUTH_EAST))
        # "Only lines of up to three marbles may be moved"
        self.assertRaises(IllegalMoveException,
                          lambda: game.move(Space.C2, Direction.EAST))
        # "Own marbles must not be moved off the board"
        self.assertRaises(IllegalMoveException, lambda: game.move(
            Space.B6, Direction.SOUTH_WEST))

        # sumito
        game.set_marble(Space.C7, Marble.WHITE)
        game.move(Space.A5, Direction.NORTH_EAST)
        assert_states([(Space.D8, Marble.WHITE), (Space.C7, Marble.BLACK), (Space.B6, Marble.BLACK),
                       (Space.A5, Marble.BLANK)])
        game.set_marble(Space.A5, Marble.WHITE)
        game.move(Space.C7, Direction.SOUTH_WEST)
        assert_states([(Space.A5, Marble.BLACK), (Space.B6,
                                                  Marble.BLACK), (Space.C7, Marble.BLANK)])
        game.set_marble(Space.C1, Marble.WHITE)
        game.move(Space.C4, Direction.WEST)
        assert_states([(Space.C1, Marble.BLACK), (Space.C2, Marble.BLACK), (Space.C3, Marble.BLACK),
                       (Space.C4, Marble.BLANK)])
        game.set_marble(Space.A1, Marble.WHITE)
        game.set_marble(Space.A2, Marble.WHITE)
        game.move(Space.A5, Direction.WEST)
        assert_states([(Space.A1, Marble.WHITE), (Space.A2, Marble.BLACK), (Space.A3, Marble.BLACK),
                       (Space.A4, Marble.BLACK), (Space.A5, Marble.BLANK)])
        game.set_marble(Space.C4, Marble.WHITE)
        # "Marbles must be pushed to an empty space or off the board"
        self.assertRaises(IllegalMoveException,
                          lambda: game.move(Space.C1, Direction.EAST))
        # "Only lines that are shorter than the player's line can be pushed"
        self.assertRaises(IllegalMoveException,
                          lambda: game.move(Space.A2, Direction.WEST))
        game.set_marble(Space.B1, Marble.WHITE)
        # "Only lines that are shorter than the player's line can be pushed"
        self.assertRaises(IllegalMoveException, lambda: game.move(
            Space.C1, Direction.SOUTH_EAST))

        # broadside
        game.move((Space.C1, Space.D2), Direction.NORTH_WEST)
        assert_states([(Space.D1, Marble.BLACK), (Space.E2, Marble.BLACK), (Space.C1, Marble.BLANK),
                       (Space.D2, Marble.BLANK)])
        # "Elements of boundaries must not be `Space.OFF`"
        self.assertRaises(IllegalMoveException, lambda: game.move(
            (Space.OFF, Space.E2), Direction.EAST))
        self.assertRaises(IllegalMoveException, lambda: game.move(
            (Space.E2, Space.OFF), Direction.EAST))
        # "Only two or three neighboring marbles may be moved with a broadside move"
        game.set_marble(Space.C4, Marble.BLACK)
        game.set_marble(Space.D5, Marble.BLANK)
        self.assertRaises(IllegalMoveException, lambda: game.move(
            (Space.E2, Space.E2), Direction.EAST))
        self.assertRaises(IllegalMoveException, lambda: game.move(
            (Space.C2, Space.C5), Direction.NORTH_EAST))
        # "The direction of a broadside move must be sideways"
        self.assertRaises(IllegalMoveException, lambda: game.move(
            (Space.D1, Space.E2), Direction.NORTH_EAST))
        # "Only own marbles may be moved"
        self.assertRaises(IllegalMoveException, lambda: game.move(
            (Space.G5, Space.G7), Direction.NORTH_EAST))
        self.assertRaises(IllegalMoveException, lambda: game.move(
            (Space.C1, Space.F3), Direction.NORTH_WEST))
        # "With a broadside move, marbles can only be moved to empty spaces"
        self.assertRaises(IllegalMoveException, lambda: game.move(
            (Space.A2, Space.A4), Direction.NORTH_EAST))
        self.assertRaises(IllegalMoveException, lambda: game.move(
            (Space.A2, Space.A4), Direction.SOUTH_EAST))
        self.assertRaises(IllegalMoveException, lambda: game.move(
            (Space.C2, Space.C3), Direction.SOUTH_WEST))

    def test_generate_random_move(self):
        game = Game()
        inline, broadside = 0, 0
        for _ in range(1000):
            move = game.generate_random_move()
            self.assertTrue(game.is_valid_move(*move))
            game.move(*move)
            if isinstance(move[0], tuple):
                broadside += 1
            else:
                inline += 1
        print(inline, broadside)

    def test_generate_legal_moves(self):
        """Test `abalone_engine.game.Game.generate_legal_moves` including\
        `abalone_engine.game.Game.generate_own_marble_lines`"""

        game = Game()
        legal_moves = list(game.generate_legal_moves())

        self.assertIn((Space.A1, Direction.NORTH_EAST), legal_moves)
        self.assertIn((Space.A1, Direction.NORTH_WEST), legal_moves)
        self.assertIn((Space.A2, Direction.NORTH_EAST), legal_moves)
        self.assertIn(
            ((Space.B1, Space.B2), Direction.NORTH_WEST), legal_moves)
        self.assertNotIn((Space.A1, Direction.SOUTH_EAST), legal_moves)
        self.assertNotIn((Space.B1, Direction.SOUTH_EAST), legal_moves)
        self.assertNotIn((Space.C1, Direction.SOUTH_EAST), legal_moves)
        self.assertNotIn((Space.D1, Direction.EAST), legal_moves)
        self.assertNotIn((Space.I5, Direction.SOUTH_WEST), legal_moves)
        self.assertNotIn(
            ((Space.C3, Space.C5), Direction.SOUTH_EAST), legal_moves)

        game.switch_player()
        legal_moves = list(game.generate_legal_moves())

        self.assertIn((Space.G5, Direction.EAST), legal_moves)
        self.assertIn((Space.I9, Direction.SOUTH_EAST), legal_moves)
        self.assertIn(
            ((Space.G5, Space.G7), Direction.SOUTH_WEST), legal_moves)
        self.assertNotIn((Space.I5, Direction.NORTH_EAST), legal_moves)
        self.assertNotIn(
            ((Space.C3, Space.C5), Direction.NORTH_WEST), legal_moves)

    def test_new_generate_legal_moves(self):
        """Test `abalone_engine.game.Game.generate_legal_moves` including\
        `abalone_engine.game.Game.generate_own_marble_lines`"""
        def validate(game: Game, marbles: Union[Space, Tuple[Space, Space]] = None, direction: Direction = None):
            if marbles and direction:
                game.move(marbles, direction)
                game.switch_player()

            legal_moves = list(game.old_generate_legal_moves())
            new_legal_moves = list(game.generate_legal_moves())
            self.assertEqual(len(legal_moves), len(new_legal_moves))
            for move in legal_moves:
                self.assertIn(move, new_legal_moves)

        game = Game()

        N = 10
        total_old = 0
        total_new = 0
        for i in range(N):
            old_start = time.time()
            legal_moves = list(game.old_generate_legal_moves())
            old_end = time.time()
            new_start = time.time()
            new_legal_moves = list(game.generate_legal_moves())
            new_end = time.time()
            total_old += old_end - old_start
            total_new += new_end - new_start

        total_old_avg = total_old / N
        total_new_avg = total_new / N
        print(f'old: {total_old_avg}')
        print(f'new: {total_new_avg}')
        print(f'ratio: {total_new_avg/total_old_avg}')
        self.assertGreater(total_old, total_new_avg)

        validate(game)
        validate(game, Space.A1, Direction.NORTH_EAST)
        validate(game, Space.G5, Direction.SOUTH_EAST)
        validate(game, Space.D4, Direction.NORTH_WEST)

    def test_s_is_over(self):
        score = Game.s_score(self.TEST_BOARD)
        self.assertEqual(score, (14, 14))
        self.assertFalse(Game.s_is_over(score))
        finished_game = np.copy(self.TEST_BOARD)
        finished_game[8] = finished_game[8] * 0
        finished_game[7, 0] = 0
        new_score = Game.s_score(finished_game)
        winner = Game.s_winner(new_score)
        self.assertEqual(new_score, (8, 14))
        self.assertEqual(winner, Player.WHITE)
        self.assertTrue(Game.s_is_over(new_score))


class TestMove:
    def test_from_standard_conversion(self):
        moves = [
            "a5NW",
        ]
        for move in moves:
            new_move = Move.from_standard(move)

    def test_full_standard_conversion(self):
        moves = [
            Move(
                first=Space.A1,
                direction=Direction.NORTH_EAST
            ),
            Move(
                first=Space.C3,
                second=Space.C5,
                direction=Direction.NORTH_WEST
            ),
        ]
        for move in moves:
            new_move = Move.from_standard(move.to_standard())
            assert new_move.first == move.first
            assert new_move.second == move.second
            assert new_move.direction == move.direction
        # move_str = self.aba_pro.convert_move_forward()


if __name__ == '__main__':
    unittest.main()
