from dataclasses import dataclass

from abalone.hex import Cube


@dataclass
class Conversion:
    in_x: int
    in_y: int
    out_x: int
    out_y: int
    out_z: int


def test_board_to_cube():
    conversions = [
        Conversion(
            in_x=0,
            in_y=0,
            out_x=0,
            out_y=0,
            out_z=0
        ),
        Conversion(
            in_x=4,
            in_y=4,
            out_x=0,
            out_y=-4,
            out_z=4
        ),
        # Conversion(
        #     in_x=4,
        #     in_y=6,
        #     out_x=0,
        #     out_y=-6,
        #     out_z=6
        # ),
    ]
    for conversion in conversions:
        cube = Cube.from_board_array(
            conversion.in_x,
            conversion.in_y,
        )
        assert cube.x == conversion.out_x
        assert cube.y == conversion.out_y
        assert cube.z == conversion.out_z
