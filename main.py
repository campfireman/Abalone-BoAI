#!/usr/bin/env python

import os
import pickle
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from traceback import format_exc
from typing import Generator, List, Tuple, Union

from abalone_engine import utils
# import matplotlib.pyplot as plt
from abalone_engine.enums import Direction, Player, Space
from abalone_engine.game import Game, IllegalMoveException


def _merge_dict(dict1, dict2):
    for key in dict2.keys():
        dict1[key] += dict2[key]


if __name__ == '__main__':  # pragma: no cover
    # Run a game from the command line with default configuration.
    import argparse
    import importlib
    import sys

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('player_black', metavar='blk', type=str,
                        help='Python path to the Player module for black')
    parser.add_argument('player_white', metavar='wht', type=str,
                        help='Python path to the Player module for black')
    parser.add_argument('-g', '--games', type=int,
                        default=1,
                        help='Number of games to play')
    parser.add_argument('-v', '--verbose', type=bool,
                        default=False, const=True, nargs='?',
                        help='Print the game output')

    args = parser.parse_args()

    black_str = args.player_black.rsplit('.', 1)
    black = getattr(importlib.import_module(black_str[0]), black_str[1])()
    white_str = args.player_white.rsplit('.', 1)
    white = getattr(importlib.import_module(white_str[0]), white_str[1])()

    total = defaultdict(int)
    for i in range(0, args.games):
        print(f'Starting game {i}')
        start = time.time()
        game, moves_history, move_stats = Game.run_game(
            black, white, args.verbose)
        end = time.time()
        score = game.get_score()
        total_time = end-start
        utils.GameStats(
            name_black=str(black),
            name_white=str(white),
            score_black=score[0],
            score_white=score[1],
            total_time=total_time,
            moves=move_stats,
        ).save_pickle()
        print(f'Time to simulate: {total_time}')
        print(f'Total moves: {len(moves_history)}')

    # write_to_file(total, f'{time.time()}_data.json')
    # plt.bar(total.keys(), total.values())
    # plt.show()
