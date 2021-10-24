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

"""This module provides some functions to simplify various operations."""

import json
import os
import pickle
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pprint import pprint
from traceback import format_exc
from typing import TYPE_CHECKING, Generator, List, Tuple, Union

from abalone_engine.enums import Direction, Player, Space
from abalone_engine.exceptions import IllegalMoveException
from abalone_engine.hex import Cube

DATA_DIR = './data'


def space_to_board_indices(space: Space) -> Tuple[int, int]:
    """Returns the corresponding index for `self.board` of a given `abalone_engine.enums.Space`.

    Args:
        space: The `abalone_engine.enums.Space` for which the indices are wanted.

    Returns:
        An int tuple containing two indices for `self.board`.
    """

    xs = ['I', 'H', 'G', 'F', 'E', 'D', 'C', 'B', 'A']
    ys = ['1', '2', '3', '4', '5', '6', '7', '8', '9']

    x = xs.index(space.value[0])
    y = ys.index(space.value[1])

    # offset because lines 'F' to 'I' don't start with '1'
    if x <= 3:
        y -= 4 - x

    return x, y


def board_indices_to_space(x: int, y: int) -> Space:
    if x <= 3:
        y += 4 - x
    xs = ['I', 'H', 'G', 'F', 'E', 'D', 'C', 'B', 'A']
    ys = ['1', '2', '3', '4', '5', '6', '7', '8', '9']

    row = xs[x]
    col = ys[y]

    # offset because lines 'F' to 'I' don't start with '1'

    return getattr(Space, row + col)


def line_from_to(from_space: Space, to_space: Space) -> Union[Tuple[List[Space], Direction], Tuple[None, None]]:
    """Returns all `abalone_engine.enums.Space`s in a straight line from a given starting space to a given ending space. The\
    two bounding spaces are included. The `abalone_engine.enums.Direction` of that line is also returned.

    Example:
        ```python
        line_from_to(Space.A1, Space.D4)
        # ([Space.A1, Space.B2, Space.C3, Space.D4], Direction.NORTH_EAST)
        ```
        ```
            I · · · · ·
           H · · · · · ·
          G · · · · · · ·
         F · · · · · · · ·
        E · · · · · · · · ·
         D · · · X · · · · 9
          C · · X · · · · 8
           B · X · · · · 7
            A X · · · · 6
               1 2 3 4 5
        ```

    Args:
        from_space: The starting `abalone_engine.enums.Space`.
        to_space: The ending `abalone_engine.enums.Space`.

    Returns:
        A tuple containing a list of `abalone_engine.enums.Space`s and a `abalone_engine.enums.Direction` or `(None, None)` in case\
        no line with the given arguments is possible. The latter is also the case if the starting and ending spaces are\
        identical.

    Raises:
        Exception: Spaces must not be `abalone_engine.enums.Space.OFF`
    """
    if from_space is Space.OFF or to_space is Space.OFF:
        raise Exception('Spaces must not be `Space.OFF`')
    for direction in Direction:
        line = [from_space]
        while line[-1] is not Space.OFF:
            next_space = neighbor(line[-1], direction)
            line.append(next_space)
            if next_space is to_space:
                return line, direction
    return None, None


def new_line_from_to(from_space: Space, to_space: Space) -> Union[Tuple[List[Space], Direction], Tuple[None, None]]:
    """Returns all `abalone_engine.enums.Space`s in a straight line from a given starting space to a given ending space. The\
    two bounding spaces are included. The `abalone_engine.enums.Direction` of that line is also returned.

    Example:
        ```python
        line_from_to(Space.A1, Space.D4)
        # ([Space.A1, Space.B2, Space.C3, Space.D4], Direction.NORTH_EAST)
        ```
        ```
            I · · · · ·
           H · · · · · ·
          G · · · · · · ·
         F · · · · · · · ·
        E · · · · · · · · ·
         D · · · X · · · · 9
          C · · X · · · · 8
           B · X · · · · 7
            A X · · · · 6
               1 2 3 4 5
        ```

    Args:
        from_space: The starting `abalone_engine.enums.Space`.
        to_space: The ending `abalone_engine.enums.Space`.

    Returns:
        A tuple containing a list of `abalone_engine.enums.Space`s and a `abalone_engine.enums.Direction` or `(None, None)` in case\
        no line with the given arguments is possible. The latter is also the case if the starting and ending spaces are\
        identical.

    Raises:
        Exception: Spaces must not be `abalone_engine.enums.Space.OFF`
    """
    if from_space is Space.OFF or to_space is Space.OFF:
        raise Exception('Spaces must not be `Space.OFF`')
    from_cube = Cube.from_board_array(*space_to_board_indices(from_space))
    to_cube = Cube.from_board_array(*space_to_board_indices(to_space))
    direction = from_cube.direction(to_cube)
    line = [from_space]
    while line[-1] is not Space.OFF:
        next_space = neighbor(line[-1], direction)
        line.append(next_space)
        if next_space is to_space:
            return line, direction
    return None, None


def line_to_edge(from_space: Space, direction: Direction) -> List[Space]:
    """Returns a straight line of `abalone_engine.enums.Space`s, from a given starting space in a given\
    `abalone_engine.enums.Direction`. The line extends to the edge of the board. The starting space is included.

    Example:
        ```python
        utils.line_to_edge(Space.C4, Direction.SOUTH_EAST)
        # [Space.C4, Space.B4, Space.A4]
        ```
        ```
            I · · · · ·
           H · · · · · ·
          G · · · · · · ·
         F · · · · · · · ·
        E · · · · · · · · ·
         D · · · · · · · · 9
          C · · · X · · · 8
           B · · · X · · 7
            A · · · X · 6
               1 2 3 4 5
        ```

    Args:
        from_space: The starting `abalone_engine.enums.Space`.
        direction: The `abalone_engine.enums.Direction` of the line.

    Returns:
        A list of `abalone_engine.enums.Space`s starting with `from_space`.

    Raises:
        Exception: `from_space` must not be `abalone_engine.enums.Space.OFF`
    """
    if from_space is Space.OFF:
        raise Exception('`from_space` must not be `Space.OFF`')
    line = [from_space]
    while line[-1] is not Space.OFF:
        line.append(neighbor(line[-1], direction))
    line.pop()  # remove Space.OFF
    return line


def neighbor(space: Space, direction: Direction) -> Space:
    """Returns the neighboring `abalone_engine.enums.Space` of a given space in a given `abalone_engine.enums.Direction`.

    Example:
        ```python
        utils.neighbor(Space.B2, Direction.EAST)
        # Space.B3
        ```
        ```
            I · · · · ·
           H · · · · · ·
          G · · · · · · ·
         F · · · · · · · ·
        E · · · · · · · · ·
         D · · · · · · · · 9
          C · · · · · · · 8
           B · X N · · · 7
            A · · · · · 6
               1 2 3 4 5
        ```

    Args:
        space: The `abalone_engine.enums.Space` of which the neighbour is returned.
        direction: The `abalone_engine.enums.Direction` in which the neighbour is located.

    Returns:
        The neighboring `abalone_engine.enums.Space` of `space` in `direction`. If `space` is `abalone_engine.enums.Space.OFF`, for\
        any given `direction`, `abalone_engine.enums.Space.OFF` is returned.
    """

    if space is Space.OFF:
        return Space.OFF

    xs = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    ys = ['1', '2', '3', '4', '5', '6', '7', '8', '9']

    xi = xs.index(space.value[0])
    yi = ys.index(space.value[1])

    if direction is Direction.NORTH_EAST:
        xi += 1
        yi += 1
    elif direction is Direction.EAST:
        yi += 1
    elif direction is Direction.SOUTH_EAST:
        xi -= 1
    elif direction is Direction.SOUTH_WEST:
        xi -= 1
        yi -= 1
    elif direction is Direction.WEST:
        yi -= 1
    elif direction is Direction.NORTH_WEST:
        xi += 1

    if xi < 0 or xi >= len(xs) or yi < 0 or yi >= len(ys) or xs[xi] + ys[yi] not in Space.__members__:
        return Space.OFF

    return Space[xs[xi] + ys[yi]]


def format_move(turn: Player, move: Tuple[Union[Space, Tuple[Space, Space]], Direction], moves: int) -> str:
    """Formats a player's move as a string with a single line.

    Args:
        turn: The `Player` who performs the move
        move: The move as returned by `abalone.abstract_player.AbstractPlayer.turn`
        moves: The number of total moves made so far (not including this move)
    """
    marbles = [move[0]] if isinstance(
        move[0], Space) else line_from_to(*move[0])[0]
    marbles = map(lambda space: space.name, marbles)
    return f'{moves + 1}: {turn.name} moves {", ".join(marbles)} in direction {move[1].name}'


def game_is_over(score: Tuple[int, int]) -> bool:
    return 8 in score


def get_winner(score: Tuple[int, int]) -> Union[Player, None]:
    """Returns the winner of the game based on the current score.

    Args:
        score: The score tuple returned by `abalone.game.Game.get_score`

    Returns:
        Either the `abalone.enums.Player` who won the game or `None` if no one has won yet.
    """
    if 8 in score:
        return Player.WHITE if score[0] == 8 else Player.BLACK
    return None


def write_to_file(obj, filename):
    with open(os.path.join(DATA_DIR, filename), 'wb') as file:
        pickle.dump(obj, file)


def open_from_file(filename) -> dict:
    with open(os.path.join(DATA_DIR, filename), 'rb') as file:
        return pickle.load(file)


def write_to_file_json(obj, filename):
    with open(os.path.join(DATA_DIR, filename), 'w') as file:
        file.write(json.dumps(obj, indent=4, sort_keys=True))


@dataclass
class Stats:
    def save(self):
        path = os.path.join(DATA_DIR, self._dir)
        os.makedirs(path, exist_ok=True)
        write_to_file_json(asdict(self), os.path.join(
            self._dir, f'{time.time()}.json'))


@dataclass
class MoveStats:
    no: int
    space: str
    direction: Direction
    time: float


@dataclass
class GameStats(Stats):
    name_black: str
    name_white: str
    score_black: int
    score_white: int
    total_time: float
    moves: List[MoveStats]

    _dir = 'games'

    def save_pickle(self) -> str:
        print(f'[ ] Saving {self.name_black} vs {self.name_white}')
        filename = time.time()
        path = os.path.join(
            self._dir, f'{filename}.pickle')
        write_to_file(asdict(self), path)
        print(f'[x] Save {path}')
        return filename

    @classmethod
    def print(cls, filename):
        pprint(open_from_file(os.path.join(cls._dir, filename)))


class Storage:
    def __init__(self):
        self.zobrist = [[[0 for y in range(0, 9)]
                         for x in range(0, 9)] for p in range(0, 2)]
        self.table = {}
        self.heuristic_cache = {}
        self.children_cache = {}
        self.initialize_keys()

    def initialize_keys(self):
        '''
        Generate Zobrist hash keys
        '''
        for p in range(0, 2):
            for x in range(0, 9):
                for y in range(0, 9):
                    self.zobrist[p][x][y] = random.getrandbits(64) - 2**63

    def get_key(self, marbles: dict):
        '''
        Get the state at the key
        '''
        key = 0
        for player in marbles.keys():
            p = 0 if player == Player.BLACK.value else 1
            for x in marbles[player].keys():
                for y in marbles[player][x].keys():
                    key ^= self.zobrist[p][x][y]
        return key

    def get_tt_value(self, key: int,  marbles: dict, depth: int) -> Tuple[Tuple[Union[Space, Tuple[Space, Space]], Direction], str, float]:
        if key in self.table and self.table[key]['depth'] >= depth:
            tt_entry = self.table[key]
            return tt_entry['flag'], tt_entry['value']
        return None

    def set_tt_value(self, key: int, value: dict):
        self.table[key] = value

    def get_cache_value(self, key: int):
        return self.heuristic_cache.get(key, None)

    def set_cache_value(self, key: int, value: float):
        self.heuristic_cache[key] = value

    def get_cached_children(self, key: int):
        return self.children_cache.get(key, None)

    def set_cached_children(self, key: int, value: list):
        self.children_cache[key] = value
