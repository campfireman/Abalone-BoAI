from dataclasses import dataclass

from abalone_engine.hex import Cube


@dataclass
class Conversion:
    in_x: int
    in_y: int
    out_q: int
    out_r: int
    out_s: int


def test_board_to_cube():
    conversions = [
        # Upper left corner
        Conversion(
            in_x=0,
            in_y=0,
            out_q=0,
            out_r=-4,
            out_s=4,
        ),
        # Upper right corner
        Conversion(
            in_x=0,
            in_y=4,
            out_q=4,
            out_r=-4,
            out_s=0,
        ),
        # center
        Conversion(
            in_x=4,
            in_y=4,
            out_q=0,
            out_r=0,
            out_s=0,
        ),
        # 2 right from center
        Conversion(
            in_x=4,
            in_y=6,
            out_q=2,
            out_r=0,
            out_s=-2,
        ),
        # bottom right corner
        Conversion(
            in_x=8,
            in_y=0,
            out_q=-4,
            out_r=4,
            out_s=0,
        ),
        # bottom left corner
        Conversion(
            in_x=8,
            in_y=4,
            out_q=0,
            out_r=4,
            out_s=-4,
        ),
        # random
        Conversion(
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
