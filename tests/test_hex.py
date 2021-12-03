from dataclasses import dataclass

from abalone_engine.hex import Cube


@dataclass
class ArrayToHexConversion:
    in_x: int
    in_y: int
    out_q: int
    out_r: int
    out_s: int


def test_board_to_cube():
    conversions = [
        # Upper left corner
        ArrayToHexConversion(
            in_x=0,
            in_y=0,
            out_q=0,
            out_r=-4,
            out_s=4,
        ),
        # Upper right corner
        ArrayToHexConversion(
            in_x=0,
            in_y=4,
            out_q=4,
            out_r=-4,
            out_s=0,
        ),
        # center
        ArrayToHexConversion(
            in_x=4,
            in_y=4,
            out_q=0,
            out_r=0,
            out_s=0,
        ),
        # 2 right from center
        ArrayToHexConversion(
            in_x=4,
            in_y=6,
            out_q=2,
            out_r=0,
            out_s=-2,
        ),
        # bottom right corner
        ArrayToHexConversion(
            in_x=8,
            in_y=0,
            out_q=-4,
            out_r=4,
            out_s=0,
        ),
        # bottom left corner
        ArrayToHexConversion(
            in_x=8,
            in_y=4,
            out_q=0,
            out_r=4,
            out_s=-4,
        ),
        # random
        ArrayToHexConversion(
            in_x=5,
            in_y=5,
            out_q=1,
            out_r=1,
            out_s=-2,
        ),
    ]
    for conversion in conversions:
        cube = Cube.from_board_array(
            conversion.in_x,
            conversion.in_y,
        )
        assert cube.q == conversion.out_q
        assert cube.r == conversion.out_r
        assert cube.s == conversion.out_s

        x, y = cube.to_board_array()
        assert x == conversion.in_x
        assert y == conversion.in_y


@dataclass
class HexRotation:
    degrees: int
    in_q: int
    in_r: int
    in_s: int
    out_q: int
    out_r: int
    out_s: int


def test_rotate():
    rotations = [
        # full rotation
        HexRotation(
            degrees=60,
            in_q=4,
            in_r=-4,
            in_s=0,
            out_q=4,
            out_r=0,
            out_s=-4,
        ),
        HexRotation(
            degrees=120,
            in_q=4,
            in_r=-4,
            in_s=0,
            out_q=0,
            out_r=4,
            out_s=-4,
        ),
        HexRotation(
            degrees=180,
            in_q=4,
            in_r=-4,
            in_s=0,
            out_q=-4,
            out_r=4,
            out_s=0,
        ),
        HexRotation(
            degrees=240,
            in_q=4,
            in_r=-4,
            in_s=0,
            out_q=-4,
            out_r=0,
            out_s=4,
        ),
        HexRotation(
            degrees=300,
            in_q=4,
            in_r=-4,
            in_s=0,
            out_q=0,
            out_r=-4,
            out_s=4,
        ),
        HexRotation(
            degrees=360,
            in_q=4,
            in_r=-4,
            in_s=0,
            out_q=4,
            out_r=-4,
            out_s=0,
        ),
        # random points
        HexRotation(
            degrees=180,
            in_q=3,
            in_r=-1,
            in_s=-2,
            out_q=-3,
            out_r=1,
            out_s=2,
        ),
        HexRotation(
            degrees=240,
            in_q=1,
            in_r=-1,
            in_s=0,
            out_q=-1,
            out_r=0,
            out_s=1,
        ),
    ]
    for rotation in rotations:
        before = Cube(rotation.in_q, rotation.in_r, rotation.in_s)
        after = before.rotate(rotation.degrees)
        assert after.q == rotation.out_q
        assert after.r == rotation.out_r
        assert after.s == rotation.out_s
