import asyncio
import json
import os
import subprocess
import sys
from abc import ABC, abstractmethod
from typing import List, Tuple, Union
from uuid import uuid4

from abalone_engine.enums import Direction, Player, Space
from abalone_engine.game import Game, Move
from abalone_engine.players import AbstractPlayer


class PipePlayer(AbstractPlayer, ABC):
    SENDING_PIPE = '/tmp/abalone_sending'
    RECIEVING_PIPE = '/tmp/abalone_recieving'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sending_pipe = self.create_pipe_name(self.SENDING_PIPE)
        self.recieving_pipe = self.create_pipe_name(self.RECIEVING_PIPE)

        if not os.path.exists(self.sending_pipe):
            os.mkfifo(self.sending_pipe)
        if not os.path.exists(self.recieving_pipe):
            os.mkfifo(self.recieving_pipe)

    def create_pipe_name(self, root: str) -> str:
        return f'{root}_{str(uuid4())}'

    def read_move(self, pipe_path: str = None) -> str:
        if pipe_path is None:
            pipe_path = self.recieving_pipe
        with open(pipe_path, 'r') as pipe:
            message = pipe.read().strip()
            print(f'Python read: {message}')
            return message

    def send_move(self, move: str, pipe_path: str = None):
        if pipe_path is None:
            pipe_path = self.sending_pipe
        with open(pipe_path, 'w') as pipe:
            pipe.write(move)
        print(f'Python sent: {move}')

    @abstractmethod
    def spawn_remote_player(self):
        '''Loads the remote playing process'''

    def convert_move_forward(self, move: Tuple[Union[Space, Tuple[Space, Space]], Direction]) -> str:
        if type(move[0]) == tuple:
            return Move(
                first=move[0][0],
                second=move[0][1],
                direction=move[1]
            ).to_standard()
        return Move(
            first=move[0],
            direction=move[1]
        ).to_standard()

    def convert_move_backward(self, move: str) -> Tuple[Union[Space, Tuple[Space, Space]], Direction]:
        converted = Move.from_standard(move)
        if converted.second:
            return ((converted.first, converted.second), converted.direction)
        return (converted.first, converted.direction)

    def turn(self, game: Game, moves_history: List[Tuple[Union[Space, Tuple[Space, Space]], Direction]]) \
            -> Tuple[Union[Space, Tuple[Space, Space]], Direction]:
        if len(moves_history) > 0:
            self.send_move(self.convert_move_forward(moves_history[-1]))
        return self.convert_move_backward(self.read_move())

    def __del__(self):
        # delete pipes
        if os.path.exists(self.sending_pipe):
            os.unlink(self.sending_pipe)
        if os.path.exists(self.recieving_pipe):
            os.unlink(self.recieving_pipe)


class AbaProPlayer(PipePlayer):
    JSON_ALGO = {
        "@class": "computer",
        "name": "Player",
        "strategy": {
                "@class": "minimax",
                "evaluator": {
                    "@class": "evaluator",
                    "abaPro": False,
                    "considerEnemyPosition": True,
                    "coherenceWeight": 3,
                    "distanceFromCenterWeight": 8,
                    "formationBreakWeight": 10,
                    "marblesConqueredWeight": 800,
                    "immediateMarbleCapWeight": 0,
                    "singleMarbleCapWeight": 30,
                    "doubleMarbleCapWeight": 50
                },
            "miniBuilder": {
                    "@class": "builder",
                    "dfs": True,
                    "depthBoundIddfs": False,
                    "depth": 3,
                    "timeBoundIddfs": True,
                    "time": 15,
                    "hashing": True,
                    "windowNarrowing": False,
                    "evaluateSorting": True,
                    "evaluateSortingMinDepth": 1,
                    "evaluateSortingMaxDepth": 2,
                    "historyHeuristicSorting": True,
                    "historyHeuristicSortingMinDepth": 3,
                    "historyHeuristicSortingMaxDepth": 4,
                    "marbleOrdering": True,
                    "marbleOrderingMinDepth": 5,
                    "marbleOrderingMaxDepth": 5,
                    "iterationSorting": True,
                    "iterationSortingMinDepth": 3,
                    "iterationSortingMaxDepth": 5
            }
        }
    }
    JSON_REMOTE = {
        "@class": "remote",
        "name": "remote player",
    }
    SETTINGS_FOLDER = '/tmp/'

    def __init__(self, *args, depth: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.settings_path = os.path.join(
            self.SETTINGS_FOLDER, "settings.json")
        self.jar_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(
            __file__))), 'lib', 'abalone-latest-jar-with-dependencies.jar')

        # update JSON features
        self.JSON_ALGO['strategy']['miniBuilder']['depth'] = depth
        self.JSON_REMOTE['recievingPipePath'] = self.sending_pipe
        self.JSON_REMOTE['sendingPipePath'] = self.recieving_pipe

        if self.player == Player.WHITE:
            settings = [self.JSON_REMOTE, self.JSON_ALGO]
        else:
            settings = [self.JSON_ALGO, self.JSON_REMOTE]
        with open(self.settings_path, 'w') as file:
            file.write(json.dumps(settings))
        self.spawn_remote_player()

    async def run(self, cmd):
        self.proc = await asyncio.create_subprocess_shell(cmd)

    def spawn_remote_player(self):
        asyncio.run(
            self.run(f'java -jar {self.jar_path} --players {self.settings_path}'))

    def __del__(self):
        super().__del__()
        # delete settings file
        os.unlink(self.settings_path)
