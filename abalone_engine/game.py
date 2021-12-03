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

"""This module serves the representation of game states and the performing of game moves."""
from __future__ import annotations

import re
import time
from collections import defaultdict
from copy import deepcopy
from random import choice, randrange
from typing import Dict, Generator, List, Optional, Tuple, Union

import colorama
import numpy as np
import numpy.typing as npt
from colorama import Style

from abalone_engine.enums import (Direction, InitialPosition, Marble, Player,
                                  Space)
from abalone_engine.exceptions import IllegalMoveException
from abalone_engine.hex import Cube
from abalone_engine.utils import (GameStats, MoveStats, board_indices_to_space,
                                  format_exc, format_move, get_winner,
                                  line_from_to, line_to_edge, neighbor,
                                  new_line_from_to, space_to_board_indices)

colorama.init(autoreset=True)
TOTAL_NUM_MARBLES = 14
TAKEN_MARBLES_TO_WIN = 6


def _opposite_direction(direction: Direction):
    if direction is direction.NORTH_EAST:
        return direction.SOUTH_WEST
    elif direction is direction.EAST:
        return Direction.WEST
    elif direction is direction.SOUTH_EAST:
        return Direction.NORTH_WEST
    elif direction is direction.SOUTH_WEST:
        return Direction.NORTH_EAST
    elif direction is Direction.WEST:
        return Direction.EAST
    elif direction is Direction.NORTH_WEST:
        return Direction.SOUTH_EAST


def _marble_of_player(player: Player) -> Marble:
    """Returns the corresponding `abalone_engine.enums.Marble` for a given `abalone_engine.enums.Player`.

    Args:
        player: The `abalone_engine.enums.Player` whose `abalone_engine.enums.Marble` is wanted.

    Returns:
        The `abalone_engine.enums.Marble` which belongs to `player`.
    """

    return Marble.WHITE if player is Player.WHITE else Marble.BLACK


class Move:
    def __init__(self, first: Space, direction: Direction, second: Optional[Space] = None):
        self.first = first
        self.direction = direction
        self.second = second

    @classmethod
    def from_standard(cls, move: str) -> Move:
        match = re.fullmatch(
            r'([A-Ia-i][1-9]){1}([A-Ia-i][1-9]){0,1}((NE)|(E)|(SE)|(SE)|(SW)|(W)|(NW)){1}', move)
        if match is None:
            raise ValueError(f'Move {move} has incorrent format.')
        string = match.group(0)
        first = cls.space_str_to_enum(string[0:2].upper())
        if len(string) == 6 or len(string) == 5:
            second = cls.space_str_to_enum(string[2:4].upper())
            direction = cls.dir_str_to_enum(string[4:])
        else:
            second = None
            direction = cls.dir_str_to_enum(string[2:])

        return cls(
            first=first,
            second=second,
            direction=direction,
        )

    @classmethod
    def from_original(cls, move: Tuple[Union[Space, Tuple[Space, Space]], Direction]) -> Move:
        marbles = move[0]
        if type(marbles) is tuple:
            return cls(
                first=marbles[0],
                second=marbles[1],
                direction=move[1]
            )
        return cls(
            first=marbles,
            direction=move[1]
        )

    @staticmethod
    def space_str_to_enum(space: str) -> Space:
        return Space(tuple(list(space)))

    @staticmethod
    def dir_str_to_enum(direction: str) -> Direction:
        return Direction(direction)

    def to_standard(self) -> str:
        first = ''.join(self.first.value)
        second = ''.join(self.second.value) if self.second else ""
        if first > second:
            first, second = second, first
        direction = self.direction.value
        return f'{first}{second}{direction}'

    def to_original(self) -> Tuple[Union[Tuple[Space, Space], Space], Direction]:
        if self.second:
            return ((self.first, self.second), self.direction)
        return (self.first, self.direction)

    def is_inline(self) -> bool:
        return self.second is None

    def rotate(self, degrees: int) -> Move:
        for name, space in {'first': self.first, 'second': self.second}.items():
            if space is None:
                continue
            x, y = Cube.from_board_array(
                *space_to_board_indices(space)).rotate(degrees).to_board_array()
            new_space = board_indices_to_space(x, y)
            setattr(self, name, new_space)
        vec = Cube.DIRECTIONS_TO_CUBE[self.direction]
        rotated_direction_cube = Cube(vec[0], vec[1], vec[2]).rotate(degrees)
        self.direction = Cube.CUBE_TO_DIRECTIONS[(
            rotated_direction_cube.q,
            rotated_direction_cube.r,
            rotated_direction_cube.s,
        )]
        return self

    def reflect(self, axis: str) -> Move:
        for name, space in {'first': self.first, 'second': self.second}.items():
            if space is None:
                continue
            cube = Cube.from_board_array(
                *space_to_board_indices(space))
            x, y = getattr(cube, f'reflect_{axis}')().to_board_array()
            new_space = board_indices_to_space(x, y)
            setattr(self, name, new_space)

        coords = Cube.DIRECTIONS_TO_CUBE[self.direction]
        cube = Cube(coords[0], coords[1], coords[2])
        reflected_cube_direction = getattr(cube, f'reflect_{axis}')()
        self.direction = Cube.CUBE_TO_DIRECTIONS[(
            reflected_cube_direction.q,
            reflected_cube_direction.r,
            reflected_cube_direction.s,
        )]
        return self


class Game:
    """Represents the mutable state of an Abalone game."""

    def __init__(self, initial_position: InitialPosition = InitialPosition.DEFAULT, first_turn: Player = Player.BLACK):
        self.board = deepcopy(initial_position.value)
        self.turn = first_turn
        self.marbles = self.init_marbles()

    @classmethod
    def from_array(cls, board: npt.NDArray, player: int) -> Game:
        game = cls(InitialPosition.EMPTY, first_turn=Player(player))
        for x in range(0, len(board)):
            for y in range(0, len(board[x])):
                marble = Marble(board[x][y])
                if marble is Marble.BLANK:
                    continue
                if x < 4:
                    y = y - (4 - x)
                game.board[x][y] = marble
        game.marbles = game.init_marbles()
        return game

    def __str__(self) -> str:  # pragma: no cover
        board_lines = list(
            map(lambda line: ' '.join(map(str, line)), self.board))
        string = ''
        string += Style.DIM + '    I ' + Style.NORMAL + board_lines[0] + '\n'
        string += Style.DIM + '   H ' + Style.NORMAL + board_lines[1] + '\n'
        string += Style.DIM + '  G ' + Style.NORMAL + board_lines[2] + '\n'
        string += Style.DIM + ' F ' + Style.NORMAL + board_lines[3] + '\n'
        string += Style.DIM + 'E ' + Style.NORMAL + board_lines[4] + '\n'
        string += Style.DIM + ' D ' + Style.NORMAL + \
            board_lines[5] + Style.DIM + ' 9\n' + Style.NORMAL
        string += Style.DIM + '  C ' + Style.NORMAL + \
            board_lines[6] + Style.DIM + ' 8\n' + Style.NORMAL
        string += Style.DIM + '   B ' + Style.NORMAL + \
            board_lines[7] + Style.DIM + ' 7\n' + Style.NORMAL
        string += Style.DIM + '    A ' + Style.NORMAL + \
            board_lines[8] + Style.DIM + ' 6\n' + Style.NORMAL
        string += Style.DIM + '       1 2 3 4 5' + Style.NORMAL
        return string

    def init_marbles(self) -> Dict[Dict, Dict]:
        marbles = {-1: defaultdict(dict), 1: defaultdict(dict)}
        for space in Space:
            if space is Space.OFF:
                continue
            x, y = space_to_board_indices(space)
            marble = self.board[x][y]
            if marble is not Marble.BLANK:
                marbles[marble.value][x][y] = marble
        return marbles

    def is_over(self) -> bool:
        return 8 in self.get_score()

    def get_winner(self) -> bool:
        score = self.get_score()
        if 8 in score:
            return Player.WHITE if score[0] == 8 else Player.BLACK
        return None

    def not_in_turn_player(self) -> Player:
        """Gets the `abalone_engine.enums.Player` who is currently *not* in turn. Returns `abalone_engine.enums.Player.WHITE` when\
        `abalone_engine.enums.Player.BLACK` is in turn and vice versa. This player is commonly referred to as "opponent" in\
        other places.

        Returns:
            The `abalone_engine.enums.Player` not in turn.
        """

        return Player.BLACK if self.turn is Player.WHITE else Player.WHITE

    def switch_player(self) -> None:
        """Switches the player whose turn it is."""
        self.turn = self.not_in_turn_player()

    def canonical_board(self) -> npt.NDArray:
        """creates a 9x9 array from current internal representation
           0 1 2 3 4 5 6 7 8 
        0          o o o o o
        1        o o o o o o
        2      · · o o o · ·
        3    · · · · · · · ·
        4  · · · · · · · · ·
        5  · · · · · · · · 
        6  · · ● ● ● · ·   
        7  ● ● ● ● ● ●    
        8  ● ● ● ● ●     

        Returns:
            List[List[int]]: canonical board representation where role is switched dependending on the player in turn
        """
        board = np.zeros((9, 9), dtype='int8')
        # invert colors
        black_val = 1 if self.turn is Player.BLACK else -1
        white_val = 1 if self.turn is Player.WHITE else -1
        for p in (Player.BLACK.value, Player.WHITE.value):
            for x in self.marbles[p].keys():
                for y in self.marbles[p][x].keys():
                    marble = self.board[x][y]
                    if x < 4:
                        y = y + (4 - x)
                    if marble is Marble.BLACK:
                        board[x][y] = black_val
                    else:
                        board[x][y] = white_val
        return board

    def to_array(self) -> List[List[int]]:
        board = np.zeros((9, 9), dtype='int8')
        for p in (Player.BLACK.value, Player.WHITE.value):
            for x in self.marbles[p].keys():
                for y in self.marbles[p][x].keys():
                    marble = self.board[x][y]
                    if x < 4:
                        y = y + (4 - x)
                    board[x][y] = marble.value
        return board

    def to_rotated_array(self, degrees: int, clockwise: bool = True) -> List[List[int]]:
        board = np.zeros((9, 9), dtype='int8')
        for p in (Player.BLACK.value, Player.WHITE.value):
            for x in self.marbles[p].keys():
                for y in self.marbles[p][x].keys():
                    marble = self.board[x][y]
                    new_x, new_y = Cube.from_board_array(
                        x, y).rotate(degrees, clockwise).to_board_array()
                    if x < 4:
                        y = y + (4 - x)
                    board[new_x][new_y] = marble.value
        return board

    def set_marble(self, space: Space, marble: Marble) -> None:
        """Updates the state of a `abalone_engine.enums.Space` on the board.

        Args:
            space: The `abalone_engine.enums.Space` to be updated.
            marble: The new state of `space` of type `abalone_engine.enums.Marble`

        Raises:
            Exception: Cannot set state of `abalone_engine.enums.Space.OFF`
        """

        if space is Space.OFF:
            raise Exception('Cannot set state of `Space.OFF`')

        x, y = space_to_board_indices(space)
        prev_marble = self.board[x][y]
        if prev_marble != Marble.BLANK:
            del self.marbles[prev_marble.value][x][y]
            if len(list(self.marbles[prev_marble.value][x].items())) == 0:
                del self.marbles[prev_marble.value][x]
        if marble != Marble.BLANK:
            self.marbles[marble.value][x][y] = marble
        self.board[x][y] = marble

    def get_marble(self, space: Space) -> Marble:
        """Returns the state of a `abalone_engine.enums.Space`.

        Args:
            space: The `abalone_engine.enums.Space` of which the state is to be returned.

        Returns:
            A `abalone_engine.enums.Marble` representing the current state of `space`.

        Raises:
            Exception: Cannot get state of `abalone_engine.enums.Space.OFF`
        """

        if space is Space.OFF:
            raise Exception('Cannot get state of `Space.OFF`')

        x, y = space_to_board_indices(space)

        return self.board[x][y]

    def old_get_score(self) -> Tuple[int, int]:
        """Counts how many marbles the players still have on the board.

        Returns:
            A tuple with the number of marbles of black and white, in that order.
        """
        black = 0
        white = 0
        for row in self.board:
            for space in row:
                if space is Marble.BLACK:
                    black += 1
                elif space is Marble.WHITE:
                    white += 1
        return black, white

    def get_score(self) -> Tuple[int, int]:
        """Counts how many marbles the players still have on the board.

        Returns:
            A tuple with the number of marbles of black and white, in that order.
        """
        score = {
            Player.BLACK.value: 0,
            Player.WHITE.value: 0,
        }
        for player in (Player.BLACK.value, Player.WHITE.value):
            for row in self.marbles[player].values():
                for col in row.values():
                    score[player] += 1

        return score[Player.BLACK.value], score[Player.WHITE.value]

    def _inline_marbles_nums(self, line: List[Space]) -> Tuple[int, int]:
        """Counts the number of own and enemy marbles that are in the given line. First the directly adjacent marbles\
        of the player whose turn it is are counted and then the subsequent directly adjacent marbles of the opponent.\
        Therefore only the marbles that are relevant for an inline move are counted. This method serves as an\
        helper method for `abalone_engine.game.Game.move_inline`.

        Args:
            line: A list of `abalone_engine.enums.Space`s that are in a straight line.

        Returns:
            A tuple with the number of 1. own marbles and 2. opponent marbles, according to the counting method\
            described above.
        """
        own_marbles_num = 0
        while own_marbles_num < len(line) and self.get_marble(line[own_marbles_num]) is _marble_of_player(self.turn):
            own_marbles_num += 1
        opp_marbles_num = 0
        while opp_marbles_num + own_marbles_num < len(line) and self.get_marble(
                line[opp_marbles_num + own_marbles_num]) is _marble_of_player(self.not_in_turn_player()):
            opp_marbles_num += 1
        return own_marbles_num, opp_marbles_num

    def move_inline(self, caboose: Space, direction: Direction, persistent: bool = True) -> None:
        """Performs an inline move. An inline move is denoted by the trailing marble ("caboose") of a straight line of\
        marbles. Marbles of the opponent can only be pushed with an inline move (as opposed to a broadside move). This\
        is possible if the opponent's marbles are directly in front of the line of the player's own marbles, and only\
        if the opponent's marbles are outnumbered ("sumito") and are moved to an empty space or off the board.

        Args:
            caboose: The `abalone_engine.enums.Space` of the trailing marble of a straight line of up to three marbles.
            direction: The `abalone_engine.enums.Direction` of movement.

        Raises:
            IllegalMoveException: Only own marbles may be moved
            IllegalMoveException: Only lines of up to three marbles may be moved
            IllegalMoveException: Own marbles must not be moved off the board
            IllegalMoveException: Only lines that are shorter than the player's line can be pushed
            IllegalMoveException: Marbles must be pushed to an empty space or off the board
        """

        if self.get_marble(caboose) is not _marble_of_player(self.turn):
            raise IllegalMoveException('Only own marbles may be moved')

        line = line_to_edge(caboose, direction)
        own_marbles_num, opp_marbles_num = self._inline_marbles_nums(line)

        if own_marbles_num > 3:
            raise IllegalMoveException(
                'Only lines of up to three marbles may be moved')

        if own_marbles_num == len(line):
            raise IllegalMoveException(
                'Own marbles must not be moved off the board')

        # sumito
        if opp_marbles_num > 0:
            if opp_marbles_num >= own_marbles_num:
                raise IllegalMoveException(
                    'Only lines that are shorter than the player\'s line can be pushed')
            push_to = neighbor(
                line[own_marbles_num + opp_marbles_num - 1], direction)
            if push_to is not Space.OFF:
                if self.get_marble(push_to) is _marble_of_player(self.turn):
                    raise IllegalMoveException(
                        'Marbles must be pushed to an empty space or off the board')
                if persistent:
                    self.set_marble(push_to, _marble_of_player(
                        self.not_in_turn_player()))
        if persistent:
            self.set_marble(line[own_marbles_num],
                            _marble_of_player(self.turn))
            self.set_marble(caboose, Marble.BLANK)

    def move_broadside(self, boundaries: Tuple[Space, Space], direction: Direction, persistent: bool = True) -> None:
        """Performs a broadside move. With a broadside move a line of adjacent marbles is moved sideways into empty\
        spaces. However, it is not possible to push the opponent's marbles. A broadside move is denoted by the two\
        outermost `abalone_engine.enums.Space`s of the line to be moved and the `abalone_engine.enums.Direction` of movement. With a\
        broadside move two or three marbles can be moved, i.e. the two boundary marbles are either direct neighbors or\
        there is exactly one marble in between.

        Args:
            boundaries: A tuple of the two outermost `abalone_engine.enums.Space`s of a line of two or three marbles.
            direction: The `abalone_engine.enums.Direction` of movement.

        Raises:
            IllegalMoveException: Elements of boundaries must not be `abalone_engine.enums.Space.OFF`
            IllegalMoveException: Only two or three neighboring marbles may be moved with a broadside move
            IllegalMoveException: The direction of a broadside move must be sideways
            IllegalMoveException: Only own marbles may be moved
            IllegalMoveException: With a broadside move, marbles can only be moved to empty spaces
        """
        if boundaries[0] is Space.OFF or boundaries[1] is Space.OFF:
            raise IllegalMoveException(
                'Elements of boundaries must not be `Space.OFF`')
        marbles, direction1 = line_from_to(boundaries[0], boundaries[1])
        if marbles is None or not (len(marbles) == 2 or len(marbles) == 3):
            raise IllegalMoveException(
                'Only two or three neighboring marbles may be moved with a broadside move')
        _, direction2 = line_from_to(boundaries[1], boundaries[0])
        if direction is direction1 or direction is direction2:
            raise IllegalMoveException(
                'The direction of a broadside move must be sideways')
        for marble in marbles:
            if self.get_marble(marble) is not _marble_of_player(self.turn):
                raise IllegalMoveException('Only own marbles may be moved')
            destination_space = neighbor(marble, direction)
            if destination_space is Space.OFF or self.get_marble(destination_space) is not Marble.BLANK:
                raise IllegalMoveException(
                    'With a broadside move, marbles can only be moved to empty spaces')
        if persistent:
            for marble in marbles:
                self.set_marble(marble, Marble.BLANK)
                self.set_marble(neighbor(marble, direction),
                                _marble_of_player(self.turn))

    def move(self, marbles: Union[Space, Tuple[Space, Space]], direction: Direction, persistent: bool = True) -> None:
        """Performs either an inline or a broadside move, depending on the arguments passed, by calling the according\
        method (`abalone_engine.game.Game.move_inline` or `abalone_engine.game.Game.move_broadside`).

        Args:
            marbles: The `abalone_engine.enums.Space`s with the marbles to be moved. Either a single space for an inline move\
                or a tuple of two spaces for a broadside move, in accordance with the parameters of\
                `abalone_engine.game.Game.move_inline` resp. `abalone_engine.game.Game.move_broadside`.
            direction: The `abalone_engine.enums.Direction` of movement.

        Raises:
            Exception: Invalid arguments
        """
        if isinstance(marbles, Space):
            self.move_inline(marbles, direction, persistent=persistent)
        elif isinstance(marbles, tuple) and isinstance(marbles[0], Space) and isinstance(marbles[1], Space):
            self.move_broadside(marbles, direction, persistent=persistent)
        else:  # pragma: no cover
            # This exception should only be raised if the arguments are not passed according to the type hints. It is
            # only there to prevent a silent failure in such a case.
            raise Exception('Invalid arguments')

    def standard_move(self, move: str) -> None:
        move = Move.from_standard(move).to_original()
        self.move(move[0], move[1])

    def generate_random_move(self) -> Tuple[Union[Tuple[Space, Space], Space], Direction]:
        directions = [d for d in Direction]
        while (True):
            own_marbles = self.marbles[self.turn.value]
            x, selected_row = choice(list(own_marbles.items()))
            y, marble = choice(list(selected_row.items()))
            space = board_indices_to_space(x, y)
            marbles = space
            space2 = None
            direction = choice(directions)
            neighbor1 = neighbor(space, direction)
            if neighbor1 is not Space.OFF and self.get_marble(neighbor1) is _marble_of_player(self.turn) and choice([True, False]):
                space2 = neighbor1
                neighbor2 = neighbor(neighbor1, direction)
                if neighbor2 is not Space.OFF and self.get_marble(neighbor2) is _marble_of_player(self.turn):
                    space2 = neighbor2
                marbles = (space, space2)
            if self.is_valid_move(marbles, direction):
                break
            else:
                is_valid = False
                directions_copy = directions.copy()
                directions_copy.remove(direction)
                while directions_copy:
                    i = randrange(len(directions_copy))
                    direction = directions_copy[i]
                    if self.is_valid_move(marbles, direction):
                        is_valid = True
                        break
                    else:
                        del directions_copy[i]
                if is_valid:
                    break
        return marbles, direction

    def generate_own_marble_lines(self) -> Generator[Union[Space, Tuple[Space, Space]], None, None]:
        """Generates all adjacent straight lines with up to three marbles of the player whose turn it is.

        Yields:
            Either one or two `abalone_engine.enums.Space`s according to the first parameter of `abalone_engine.game.Game.move`.
        """
        for space in Space:
            if space is Space.OFF or self.get_marble(space) is not _marble_of_player(self.turn):
                continue
            yield space
            for direction in [Direction.NORTH_WEST, Direction.NORTH_EAST, Direction.EAST]:
                neighbor1 = neighbor(space, direction)
                if neighbor1 is not Space.OFF and self.get_marble(neighbor1) is _marble_of_player(self.turn):
                    yield space, neighbor1
                    neighbor2 = neighbor(neighbor1, direction)
                    if neighbor2 is not Space.OFF and self.get_marble(neighbor2) is _marble_of_player(self.turn):
                        yield space, neighbor2

    def new_generate_own_marble_lines(self) -> Generator[Union[Space, Tuple[Space, Space]], None, None]:
        """Generates all adjacent straight lines with up to three marbles of the player whose turn it is.

        Yields:
            Either one or two `abalone_engine.enums.Space`s according to the first parameter of `abalone_engine.game.Game.move`.
        """
        for x in self.marbles[self.turn.value].keys():
            for y in self.marbles[self.turn.value][x].keys():
                space = board_indices_to_space(x, y)
                yield space
                for direction in [Direction.NORTH_WEST, Direction.NORTH_EAST, Direction.EAST]:
                    neighbor1 = neighbor(space, direction)
                    if neighbor1 is not Space.OFF and self.get_marble(neighbor1) is _marble_of_player(self.turn):
                        yield space, neighbor1
                        neighbor2 = neighbor(neighbor1, direction)
                        if neighbor2 is not Space.OFF and self.get_marble(neighbor2) is _marble_of_player(self.turn):
                            yield space, neighbor2

    def is_valid_move(self, marbles: Union[Space, Tuple[Space, Space]], direction: Direction) -> bool:
        if isinstance(marbles, Space):
            line = line_to_edge(marbles, direction)
            own_marbles_num, opp_marbles_num = self._inline_marbles_nums(line)

            if own_marbles_num > 3:
                return False

            if own_marbles_num == len(line):
                return False

            # sumito
            if opp_marbles_num > 0:
                if opp_marbles_num >= own_marbles_num:
                    return False
                push_to = neighbor(
                    line[own_marbles_num + opp_marbles_num - 1], direction)
                if push_to is not Space.OFF:
                    if self.get_marble(push_to) is _marble_of_player(self.turn):
                        return False
        elif isinstance(marbles, tuple) and isinstance(marbles[0], Space) and isinstance(marbles[1], Space):
            if marbles[0] is Space.OFF or marbles[1] is Space.OFF:
                return False
            marbles, direction1 = new_line_from_to(marbles[0], marbles[1])
            if marbles is None or not (len(marbles) == 2 or len(marbles) == 3):
                return False
            _, direction2 = new_line_from_to(marbles[1], marbles[0])
            if direction is direction1 or direction is direction2:
                return False
            for marble in marbles:
                destination_space = neighbor(marble, direction)
                if destination_space is Space.OFF or self.get_marble(destination_space) is not Marble.BLANK:
                    return False
        else:  # pragma: no cover
            # This exception should only be raised if the arguments are not passed according to the type hints. It is
            # only there to prevent a silent failure in such a case.
            return False
        return True

    def old_generate_legal_moves(self) -> Generator[Tuple[Union[Space, Tuple[Space, Space]], Direction], None, None]:
        """Generates all possible moves that the player whose turn it is can perform. The yielded values are intended\
        to be passed as arguments to `abalone_engine.game.Game.move`.

        Yields:
            A tuple of 1. either one or a tuple of two `abalone_engine.enums.Space`s and 2. a `abalone_engine.enums.Direction`
        """
        for marbles in self.new_generate_own_marble_lines():
            for direction in Direction:
                copy = deepcopy(self)
                try:
                    copy.move(marbles, direction)
                except IllegalMoveException:
                    continue
                yield marbles, direction

    def generate_legal_moves(self) -> Generator[Tuple[Union[Space, Tuple[Space, Space]], Direction], None, None]:
        """Generates all possible moves that the player whose turn it is can perform. The yielded values are intended\
        to be passed as arguments to `abalone_engine.game.Game.move`.

        Yields:
            A tuple of 1. either one or a tuple of two `abalone_engine.enums.Space`s and 2. a `abalone_engine.enums.Direction`
        """
        for marbles in self.generate_own_marble_lines():
            for direction in Direction:
                if(self.is_valid_move(marbles, direction)):
                    yield marbles, direction

    def iterate_all_marbles(self) -> Generator[Tuple[int, int, Marble]]:
        for p in (Player.BLACK.value, Player.WHITE.value):
            for x in self.marbles[p].keys():
                for y, marble in self.marbles[p][x].items():
                    yield x, y, marble

    def iterate_player_marbles(self, player: Player) -> Generator[Tuple[int, int, Marble]]:
        for x in self.marbles[player.value].keys():
            for y, marble in self.marbles[player.value][x].items():
                yield x, y, marble

    # STATIC API #
    @staticmethod
    def s_score(board: npt.NDArray) -> Tuple[int, int]:
        black_marbles = 0
        white_marbles = 0
        for x in range(0, len(board)):
            for y in range(0, len(board[x])):
                marble = board[x][y]
                if marble == Player.BLACK.value:
                    black_marbles += 1
                if marble == Player.WHITE.value:
                    white_marbles += 1
        return (black_marbles, white_marbles)

    @staticmethod
    def s_winner(score: Tuple[int, int]) -> Player:
        if score[0] == 8:
            return Player.WHITE
        if score[1] == 8:
            return Player.BLACK
        return None

    @staticmethod
    def s_is_over(score: Tuple[int, int]) -> bool:
        return 8 in score

    @staticmethod
    def s_generate_own_marble_lines(board: npt.NDArray, player: Player) -> Generator[Union[Space, Tuple[Space, Space]], None, None]:
        for space in Space:
            if space is Space.OFF or Game.s_get_marble(board, space) is not _marble_of_player(player):
                continue
            yield space
            for direction in [Direction.NORTH_WEST, Direction.NORTH_EAST, Direction.EAST]:
                neighbor1 = neighbor(space, direction)
                if neighbor1 is not Space.OFF and Game.s_get_marble(board, neighbor1) is _marble_of_player(player):
                    yield space, neighbor1
                    neighbor2 = neighbor(neighbor1, direction)
                    if neighbor2 is not Space.OFF and Game.s_get_marble(board, neighbor2) is _marble_of_player(player):
                        yield space, neighbor2

    @staticmethod
    def s_is_valid_move(board: npt.NDArray, player: Player, marbles: Union[Space, Tuple[Space, Space]], direction: Direction) -> bool:
        if isinstance(marbles, Space):
            line = line_to_edge(marbles, direction)
            own_marbles_num, opp_marbles_num = Game.s_inline_marbles_nums(
                board, line, player)

            if own_marbles_num > 3:
                return False

            if own_marbles_num == len(line):
                return False

            # sumito
            if opp_marbles_num > 0:
                if opp_marbles_num >= own_marbles_num:
                    return False
                push_to = neighbor(
                    line[own_marbles_num + opp_marbles_num - 1], direction)
                if push_to is not Space.OFF:
                    if Game.s_get_marble(board, push_to) is _marble_of_player(player):
                        return False
        elif isinstance(marbles, tuple) and isinstance(marbles[0], Space) and isinstance(marbles[1], Space):
            if marbles[0] is Space.OFF or marbles[1] is Space.OFF:
                return False
            marbles, direction1 = new_line_from_to(marbles[0], marbles[1])
            if marbles is None or not (len(marbles) == 2 or len(marbles) == 3):
                return False
            _, direction2 = new_line_from_to(marbles[1], marbles[0])
            if direction is direction1 or direction is direction2:
                return False
            for marble in marbles:
                destination_space = neighbor(marble, direction)
                if destination_space is Space.OFF or Game.s_get_marble(board, destination_space) is not Marble.BLANK:
                    return False
        else:
            return False
        return True

    @staticmethod
    def s_generate_legal_moves(board: npt.NDArray, player: int) -> Generator[Tuple[Union[Space, Tuple[Space, Space]], Direction], None, None]:
        """Generates all possible moves that the player whose turn it is can perform. The yielded values are intended\
        to be passed as arguments to `abalone_engine.game.Game.move`.

        Yields:
            A tuple of 1. either one or a tuple of two `abalone_engine.enums.Space`s and 2. a `abalone_engine.enums.Direction`
        """
        player = Player(player)
        for marbles in Game.s_generate_own_marble_lines(board, player):
            for direction in Direction:
                if(Game.s_is_valid_move(board, player, marbles, direction)):
                    yield marbles, direction

    @staticmethod
    def s_space_to_array(position: Space) -> Tuple[int, int]:
        xs = ['I', 'H', 'G', 'F', 'E', 'D', 'C', 'B', 'A']
        ys = ['1', '2', '3', '4', '5', '6', '7', '8', '9']

        x = xs.index(position.value[0])
        y = ys.index(position.value[1])

        return x, y

    @staticmethod
    def s_get_marble(board: npt.NDArray, position: Space) -> Marble:
        x, y = Game.s_space_to_array(position)
        return Marble(board[x, y])

    @staticmethod
    def s_set_marble(board: npt.NDArray, position: Space, marble: Marble) -> None:
        x, y = Game.s_space_to_array(position)
        board[x, y] = marble.value

    @staticmethod
    def s_inline_marbles_nums(board: npt.NDArray, line: List[Space], player: Player) -> Tuple[int, int]:
        own_marbles_num = 0
        while own_marbles_num < len(line) and Game.s_get_marble(board, line[own_marbles_num]) is _marble_of_player(player):
            own_marbles_num += 1
        opp_marbles_num = 0
        while opp_marbles_num + own_marbles_num < len(line) and Game.s_get_marble(board,
                                                                                  line[opp_marbles_num + own_marbles_num]) is _marble_of_player(Game.s_not_in_turn_player(player)):
            opp_marbles_num += 1
        return own_marbles_num, opp_marbles_num

    @staticmethod
    def s_not_in_turn_player(player: Player) -> Player:
        return Player.WHITE if player is Player.BLACK else Player.BLACK

    @staticmethod
    def s_move_inline(board: npt.NDArray, move: Move, player: Player) -> npt.NDArray:
        caboose = move.first
        if Game.s_get_marble(board, caboose) is not _marble_of_player(player):
            raise IllegalMoveException('Only own marbles may be moved')

        line = line_to_edge(caboose, move.direction)
        own_marbles_num, opp_marbles_num = Game.s_inline_marbles_nums(
            board, line, player)

        if own_marbles_num > 3:
            raise IllegalMoveException(
                'Only lines of up to three marbles may be moved')

        if own_marbles_num == len(line):
            raise IllegalMoveException(
                'Own marbles must not be moved off the board')

        # sumito
        if opp_marbles_num > 0:
            if opp_marbles_num >= own_marbles_num:
                raise IllegalMoveException(
                    'Only lines that are shorter than the player\'s line can be pushed')
            push_to = neighbor(
                line[own_marbles_num + opp_marbles_num - 1], move.direction)
            if push_to is not Space.OFF:
                if Game.s_get_marble(board, push_to) is _marble_of_player(player):
                    raise IllegalMoveException(
                        'Marbles must be pushed to an empty space or off the board')
                Game.s_set_marble(board, push_to, _marble_of_player(
                    Game.s_not_in_turn_player(player)))
        Game.s_set_marble(board, line[own_marbles_num],
                          _marble_of_player(player))
        Game.s_set_marble(board, caboose, Marble.BLANK)
        return board

    @staticmethod
    def s_move_broadside(board: npt.NDArray, move: Move, player: Player) -> npt.NDArray:
        if move.first is Space.OFF or move.second is Space.OFF:
            raise IllegalMoveException(
                'Elements of boundaries must not be `Space.OFF`')
        marbles, direction1 = line_from_to(move.first, move.second)
        if marbles is None or not (len(marbles) == 2 or len(marbles) == 3):
            raise IllegalMoveException(
                'Only two or three neighboring marbles may be moved with a broadside move')
        _, direction2 = line_from_to(move.second, move.first)
        if move.direction is direction1 or move.direction is direction2:
            raise IllegalMoveException(
                'The direction of a broadside move must be sideways')
        for marble in marbles:
            if Game.s_get_marble(board, marble) is not _marble_of_player(player):
                raise IllegalMoveException('Only own marbles may be moved')
            destination_space = neighbor(marble, move.direction)
            if destination_space is Space.OFF or Game.s_get_marble(board, destination_space) is not Marble.BLANK:
                raise IllegalMoveException(
                    'With a broadside move, marbles can only be moved to empty spaces')
        for marble in marbles:
            Game.s_set_marble(board, marble, Marble.BLANK)
            Game.s_set_marble(board, neighbor(marble, move.direction),
                              _marble_of_player(player))
        return board

    @staticmethod
    def s_move(board: npt.NDArray, player: int, move: Move) -> npt.NDArray:
        if move.is_inline():
            new_board = Game.s_move_inline(board, move, Player(player))
        else:
            new_board = Game.s_move_broadside(board, move, Player(player))
        return new_board

    @staticmethod
    def s_standard_move(board: npt.NDArray, player: int, move: str) -> npt.NDArray:
        return Game.s_move(board, player, Move.from_standard(move))

    @classmethod
    def run_game(cls, black: 'AbstractPlayer', white: 'AbstractPlayer', is_verbose: bool = True) \
            -> Generator[Tuple[Game, List[Tuple[Union[Space, Tuple[Space, Space]], Direction]]], None, None]:
        """Runs a game instance and prints the progress / current state at every turn.

        Args:
            black: An `abalone.abstract_player.AbstractPlayer`
            white: An `abalone.abstract_player.AbstractPlayer`
            **kwargs: These arguments are passed to `abalone.game.Game.__init__`

        """
        game = Game()
        moves_history = []
        count = defaultdict(int)
        move_stats = []

        while True:
            score = game.get_score()
            if is_verbose:
                score_str = f'BLACK {score[0]} - WHITE {score[1]}'
                print(score_str, game, '', sep='\n')

            winner = get_winner(score)
            if winner is not None and is_verbose:
                print(f'{winner.name} won!')
                break

            try:
                start = time.time()
                move = black.turn(game, moves_history) if game.turn is Player.BLACK else white.turn(
                    game, moves_history)
                end = time.time()
                if is_verbose:
                    print(f'Time to deliberate: {end-start}')
                    print(format_move(game.turn, move,
                                      len(moves_history)), end='\n\n')
                game.move(*move)
                game.switch_player()
                move_stats.append(MoveStats(
                    no=len(moves_history),
                    space=move[0],
                    direction=move[1],
                    time=end-start,
                ))
                moves_history.append(move)

            except IllegalMoveException as ex:
                if is_verbose:
                    print(
                        f'{game.turn.name}\'s tried to perform an illegal move ({ex})\n')
                break
            except:
                if is_verbose:
                    print(f'{game.turn.name}\'s move caused an exception\n')
                    print(format_exc())
                break
        return game, moves_history, move_stats

    def get_rewards(self, score) -> Tuple[float, float]:
        """Returns a reward for each player in the range of [-1, 1], also awarding partial scores
        if game had to terminate early (max turns)
        """
        if 8 in score:
            return (1.0, -1.0) if score[1] == 8 else (-1.0, 1.0)
        marbles_taken_black = (TOTAL_NUM_MARBLES -
                               score[1]) / TAKEN_MARBLES_TO_WIN
        marbles_taken_white = (TOTAL_NUM_MARBLES -
                               score[0]) / TAKEN_MARBLES_TO_WIN
        return (marbles_taken_black - marbles_taken_white, marbles_taken_white - marbles_taken_black)

    @classmethod
    def run_game_new(cls, black: 'AbstractPlayer', white: 'AbstractPlayer', is_verbose: bool = True, max_turns: int = 200):
        """Runs a game instance and prints the progress / current state at every turn.

        Args:
            black: An `abalone.abstract_player.AbstractPlayer`
            white: An `abalone.abstract_player.AbstractPlayer`
            **kwargs: These arguments are passed to `abalone.game.Game.__init__`

        """
        game = Game()
        moves_history = []
        turn = 1

        while True:
            if turn > max_turns:
                print(f'Exceeded max turns of {max_turns}.')
                break
            score = game.get_score()
            if is_verbose:
                score_str = f'BLACK {score[0]} - WHITE {score[1]}'
                print(score_str, game, '', sep='\n')

            winner = get_winner(score)
            if winner is not None and is_verbose:
                print(f'{winner.name} won!')
                break

            try:
                move = black.turn(game, moves_history) if game.turn is Player.BLACK else white.turn(
                    game, moves_history)
                if is_verbose:
                    print(format_move(game.turn, move,
                                      len(moves_history)), end='\n\n')
                game.move(*move)
                game.switch_player()

            except IllegalMoveException as ex:
                if is_verbose:
                    print(
                        f'{game.turn.name}\'s tried to perform an illegal move ({ex})\n')
                break
            except:
                if is_verbose:
                    print(f'{game.turn.name}\'s move caused an exception\n')
                    print(format_exc())
                break
            turn += 1
        return game, moves_history
