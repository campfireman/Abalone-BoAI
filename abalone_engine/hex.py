'''
Representation of hexagonal coordinate system
based on: https://www.redblobgames.com/grids/hexagons
'''
from __future__ import annotations

from typing import Tuple

from abalone_engine.enums import Direction


class Cube:
    CUBE_TO_DIRECTIONS = {
        (+1, -1, 0): Direction.NORTH_EAST,
        (+1, 0, -1): Direction.EAST,
        (0, +1, -1): Direction.SOUTH_EAST,
        (-1, +1, 0): Direction.SOUTH_WEST,
        (-1, 0, +1): Direction.WEST,
        (0, -1, +1): Direction.NORTH_WEST
    }
    DIRECTIONS_TO_CUBE = {
        Direction.NORTH_EAST: (+1, -1, 0),
        Direction.EAST: (+1, 0, -1),
        Direction.SOUTH_EAST: (0, +1, -1),
        Direction.SOUTH_WEST: (-1, +1, 0),
        Direction.WEST: (-1, 0, +1),
        Direction.NORTH_WEST: (0, -1, +1)
    }

    def __init__(self, q: int, r: int, s: int):
        self.q = q
        self.r = r
        self.s = s

    def __str__(self):
        return f'q: {self.q} r: {self.r} s: {self.s}'

    @classmethod
    def from_axial(cls, q: int, r: int) -> Cube:
        return cls(q, -q - r, r)

    def to_axial(self) -> Axial:
        return Axial(self.q, self.r)

    @classmethod
    def from_board_array(cls, x: int, y: int) -> Cube:
        if x >= 4:
            q = y - 4
        else:
            q = y - x
        r = x - 4
        s = -q - r
        return cls(q, r, s)

    def to_board_array(self) -> Tuple[int, int]:
        x = 4 + self.r
        if self.r >= 0:
            y = self.q + 4
        else:
            y = self.q + x
        return (x, y)

    def copy(self) -> Cube:
        return Cube(self.q, self.r, self.s)

    def negate(self) -> Cube:
        self.q = -self.q
        self.r = -self.r
        self.s = -self.s
        return self

    def add(self, other: Cube) -> Cube:
        self.q += other.q
        self.r += other.r
        self.s += other.s
        return self

    def sub(self, other: Cube) -> Cube:
        self.q -= other.q
        self.r -= other.r
        self.s -= other.s
        return self

    @classmethod
    def neighbor_indices(cls):
        return [
            (1, -1, 0),
            (1, 0, -1),
            (0, 1, -1),
            (-1, 1, 0),
            (-1, 0, 1),
            (0, -1, 1),
        ]

    def distance(self, other: Cube) -> int:
        return max(abs(self.q - other.q), abs(self.r - other.r), abs(self.s - other.s))

    def normalize(self) -> Cube:
        for coord, val in {'q': self.q, 'r': self.r, 's': self.s}.items():
            if val != 0:
                setattr(self, coord, int(abs(val) / val))
        return self

    def direction(self, other: Cube) -> Direction:
        vec = other.copy().sub(self).normalize()
        try:
            direction = self.CUBE_TO_DIRECTIONS[(vec.q, vec.r, vec.s)]
        except KeyError:
            print(self)
            print(other)
            raise ValueError('Vector doesn\'t have a direction')
        return direction

    def rotate(self, degrees: int) -> Cube:
        """
        rotates the cube coordinate around Cube(0, 0, 0) clockwise
        """
        if degrees % 60 != 0 or degrees > 360 or degrees < 60 or type(degrees) != int:
            raise ValueError('Invalid rotation degrees')

        if degrees == 60:
            self.q, self.r, self.s = -self.r, -self.s, -self.q
        elif degrees == 120:
            self.q, self.r, self.s = self.s, self.q, self.r
        elif degrees == 180:
            self.q, self.r, self.s = -self.q, -self.r, -self.s
        elif degrees == 240:
            self.q, self.r, self.s = self.r, self.s, self.q
        elif degrees == 300:
            self.q, self.r, self.s = -self.s, -self.q, -self.r
        # is 360 then
        else:
            pass
        return self

    def reflect(self, axis: str) -> Cube:
        return getattr(self, f'reflect_{axis}')()

    def reflect_q(self) -> Cube:
        return self.negate().reflect_qx()

    def reflect_qx(self) -> Cube:
        self.r, self.s = self.s, self.r
        return self

    def reflect_r(self) -> Cube:
        return self.negate().reflect_rx()

    def reflect_rx(self) -> Cube:
        self.q, self.s = self.s, self.q
        return self

    def reflect_s(self) -> Cube:
        return self.negate().reflect_sx()

    def reflect_sx(self) -> Cube:
        self.q, self.r = self.r, self.q
        return self


class Axial:
    def __init__(self, q: int, r: int):
        self.q = q
        self.r = r

    def to_cube(self) -> Cube:
        q = self.q
        r = self.r
        s = -q-r
        return Cube(q, r, s)
