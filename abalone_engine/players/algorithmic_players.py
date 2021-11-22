# -*- coding: utf-8 -*-
"""This module is a collection of various AI agents"""
from __future__ import annotations

import copy
import hashlib
import inspect
import math
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from math import floor
from multiprocessing import Pool
from operator import itemgetter
from random import choice
from typing import List, Tuple, Union

from abalone_engine import utils
from abalone_engine.enums import Direction, Marble, Player, Space
from abalone_engine.game import Game
from abalone_engine.hex import Cube
from abalone_engine.players import AbstractPlayer

nodes = 0
storage = utils.Storage()


def run(child, parent):
    return AlphaBetaSimple(
        child[0], parent.player, parent.depth - 1, parent.alpha, parent.beta, not parent.is_maximizer, child[1], child[2], func=parent.func).run()


def count_nodes():
    global nodes
    nodes += 1


class Heuristics:
    @classmethod
    def evaluate_move(cls, player: Player, current: Game, result: Game, marbles: Union[Space, Tuple[Space, Space]], direction: Direction) -> float:
        old_score = current.get_score()
        new_score = result.get_score()
        not_player = Player.WHITE.value if player == Player.BLACK.value else Player.BLACK.value
        if utils.game_is_over(new_score):
            w_0 = 10000
            winner = utils.get_winner(new_score)
            # return as winning or losing move
            return w_0 * 1 if winner.value == player else w_0 * -1
        enemy_selector = 1 if player == Player.BLACK.value else 0
        self_selector = 0 if player == Player.BLACK.value else 1

        captured = old_score[enemy_selector] - new_score[enemy_selector]
        lost = new_score[self_selector] - old_score[self_selector]

        multiple = 0
        attacking = 0
        if isinstance(marbles, tuple):
            c1 = Cube.from_board_array(
                *utils.space_to_board_indices(marbles[0]))
            c2 = Cube.from_board_array(
                *utils.space_to_board_indices(marbles[1]))
            multiple = c1.distance(c2)
        else:
            line = utils.line_to_edge(marbles, direction)
            own_marbles_num, opp_marbles_num = current._inline_marbles_nums(
                line)
            multiple = own_marbles_num - 1
            if opp_marbles_num > 0:
                attacking += 1
        if current.turn.value == not_player:
            multiple = -multiple
            attacking = - attacking

        # weights for the linear combination
        w_0 = 1
        w_1 = 3
        w_2 = 2
        return w_0 * multiple + w_1 * captured + w_1 * lost + w_2 * attacking


class Algorithm:
    def __str__(self):
        hashstr = hashlib.md5(inspect.getsource(
            self.__class__).encode()).hexdigest()
        return f'{self.__class__.__name__} depth = {self.depth} ({hashstr})'

    def run(self) -> Tuple[int, Union[Space, Tuple[Space, Space]], Direction]:
        raise NotImplementedError


class AlphaBetaBase(Algorithm):
    def __init__(self, game: Game, perspective: Player, depth: int = 4, alpha: float = float('-inf'), beta: float = float('inf'), is_maximizer: bool = True, marbles: Union[Space, Tuple[Space, Space]] = None, direction: Direction = None, func: function = None, initial_depth: int = 4):
        self.game = game
        self.depth = depth
        self.alpha = alpha
        self.beta = beta
        self.is_maximizer = is_maximizer
        self.marbles = marbles
        self.direction = direction
        self.player = perspective
        self.not_player = Player.BLACK.value if perspective == Player.WHITE.value else Player.WHITE.value
        self.func = func
        self.initial_depth = initial_depth
        if func:
            func()

    def _heuristic(self, game: Game) -> float:
        raise NotImplementedError

    def _order_children(self, children: List[Game]) -> List[Game]:
        return children

    def _evaluate_move(self, result: Game, marbles: Union[Space, Tuple[Space, Space]], direction: Direction) -> float:
        return 0.0

    def _end(self, result: Tuple[int, Union[Space, Tuple[Space, Space]], Direction]):
        if self.marbles is None and self.direction is None:
            return result
        return (result[0], self.marbles, self.direction)

    def _create_children(self) -> List[Tuple[Game, Union[Space, Tuple[Space, Space]], Direction, float]]:
        result = []
        for move in self.game.generate_legal_moves():
            child = copy.deepcopy(self.game)
            child.move(*move)
            child.switch_player()
            evaluation = self._evaluate_move(child, move[0], move[1])
            result.append((child, move[0], move[1], evaluation))
        result = self._order_children(result)
        # print(self.game)
        # print('---')
        # print(result)
        # print('---')
        return result

    def pre_hook(self):
        pass

    def post_hook(self, value):
        pass

    def run(self) -> Tuple[int, Union[Space, Tuple[Space, Space]], Direction]:
        pre = self.pre_hook()
        if pre != None:
            return pre

        if self.depth == 0 or utils.game_is_over(self.game.get_score()):
            return self._end((self._heuristic(self.game), None, None))
        if self.is_maximizer:
            value = (float('-inf'), None, None)
            for child in self._create_children():
                value = max(value, self.__class__(
                    child[0], self.player, self.depth - 1, self.alpha, self.beta, False, child[1], child[2], func=self.func, initial_depth=self.initial_depth).run(), key=itemgetter(0))
                self.alpha = max(self.alpha, value[0])
                if self.alpha >= self.beta:
                    break
        else:
            value = (float('inf'), None, None)
            for child in self._create_children():
                value = min(value, self.__class__(
                    child[0], self.player, self.depth - 1, self.alpha, self.beta, True, child[1], child[2], func=self.func, initial_depth=self.initial_depth).run(), key=itemgetter(0))
                self.beta = min(self.beta, value[0])
                if self.beta <= self.alpha:
                    break

        self.post_hook(value)
        return self._end(value)


class AlphaBetaSimple(AlphaBetaBase):
    def _order_children(self, children):
        children.sort(key=itemgetter(3), reverse=self.is_maximizer)
        return children

    def _evaluate_move(self, result: Game, marbles: Union[Space, Tuple[Space, Space]], direction: Direction) -> float:
        return self._heuristic(result)

    def _count_heuristics(self, game: Game) -> dict:
        result = {}
        result['sum_adjacency'] = defaultdict(int)
        result['sum_distance'] = defaultdict(int)
        center = Cube.from_board_array(4, 4)

        for player in game.marbles.keys():
            for x in game.marbles[player].keys():
                for y, current_marble in game.marbles[player][x].items():
                    c = Cube.from_board_array(x, y)
                    # adjacency
                    for neighbor in Cube.neighbor_indices():
                        n = Cube(neighbor[0], neighbor[1], neighbor[2]).add(c)
                        xn, yn = n.to_board_array()
                        if yn < 0 or xn < 0:
                            continue
                        try:
                            neighbor_marble = game.board[yn][xn]
                            if neighbor_marble == current_marble:
                                result['sum_adjacency'][current_marble.value] += 1
                        except IndexError:
                            continue
                    # distance
                    result['sum_distance'][current_marble.value] += c.distance(
                        center)
        return result

    def _heuristic(self, game: Game) -> float:
        '''
        '''
        score = game.get_score()
        if utils.game_is_over(score):
            w_0 = 100000
            winner = utils.get_winner(score)
            return w_0 * 1 if winner.value == self.player else w_0 * -1
        counts = self._count_heuristics(game)

        adjacency = counts['sum_adjacency'][self.player] - \
            counts['sum_adjacency'][self.not_player]

        distance = counts['sum_distance'][self.player] - \
            counts['sum_distance'][self.not_player]

        counter = score[0] if self.player == Player.BLACK.value else score[1]
        denominator = score[1] if self.player == Player.BLACK.value else score[0]
        marble_ratio = counter - denominator

        w_0 = 1
        w_1 = -1.5
        w_2 = 100
        heuristic = w_0 * adjacency + w_1 * distance + w_2 * marble_ratio
        # heuristic = w_2 * marble_ratio
        return heuristic


class AlphaBetaSimpleOrdering(AlphaBetaSimple):
    def _evaluate_move(self, result: Game, marbles: Union[Space, Tuple[Space, Space]], direction: Direction) -> float:
        old_score = self.game.get_score()
        new_score = result.get_score()
        if utils.game_is_over(new_score):
            w_0 = 10000
            winner = utils.get_winner(new_score)
            # return as winning or losing move
            return w_0 * 1 if winner.value == self.player else w_0 * -1
        enemy_selector = 1 if self.player == Player.BLACK.value else 0
        self_selector = 0 if self.player == Player.BLACK.value else 1

        captured = old_score[enemy_selector] - new_score[enemy_selector]
        lost = new_score[self_selector] - old_score[self_selector]

        multiple = 0
        attacking = 0
        if isinstance(marbles, tuple):
            c1 = Cube.from_board_array(*space_to_board_indices(marbles[0]))
            c2 = Cube.from_board_array(*space_to_board_indices(marbles[1]))
            multiple = c1.distance(c2)
        else:
            line = line_to_edge(marbles, direction)
            own_marbles_num, opp_marbles_num = self.game._inline_marbles_nums(
                line)
            multiple = own_marbles_num - 1
            if opp_marbles_num > 0:
                attacking += 1
        if self.game.turn.value == self.not_player:
            multiple = -multiple
            attacking = - attacking

        # weights for the linear combination
        w_0 = 1
        w_1 = 3
        w_2 = 2
        return w_0 * multiple + w_1 * captured + w_1 * lost + w_2 * attacking


class AlphaBetaAdvanced(AlphaBetaSimple):
    def __init__(self, game, perspective, depth=3, alpha=float('-inf'), beta=float('inf'), is_maximizer=True, marbles=None, direction=None, func=None):
        super().__init__(game, perspective, depth=depth, alpha=alpha, beta=beta,
                         is_maximizer=is_maximizer, marbles=marbles, direction=direction, func=func)
        self.key = storage.get_key(self.game.marbles)

    def pre_hook(self):
        result = storage.get_tt_value(self.key, self.game.marbles, self.depth)
        if result is not None:
            flag, value = result[0], result[1]
            if flag == 'lower':
                return self._end((max(self.alpha, value[0]), value[1], value[2]))
            elif flag == 'upper':
                return self._end((min(self.beta, value[0]), value[1], value[2]))
        return None

    def post_hook(self, value):
        tt_entry = {'value': None, 'flag': None, 'depth': None}
        tt_entry['value'] = value
        if value[0] <= self.alpha:
            tt_entry['flag'] = 'upper'
        elif value[0] >= self.beta:
            tt_entry['flag'] = 'lower'

        tt_entry['depth'] = self.depth
        tt_entry['board'] = self.game.marbles.copy()

        storage.set_tt_value(self.key, tt_entry)


class AlphaBetaSimpleUnordered(AlphaBetaSimple):
    def _order_children(self, children):
        return children


class AlphaBetaAdvancedFast(AlphaBetaAdvanced):
    def _order_children(self, children):
        return super()._order_children(children)[:30]


class PVS(AlphaBetaSimple):
    def _order_children(self, children):
        children = super()._order_children(children)
        # build tightening funnel
        min_cutoff = 0.4
        max_cutoff = 0.1

        if self.depth == self.initial_depth:
            cutoff = min_cutoff
        elif self.depth == 1:
            cutoff = max_cutoff
        else:
            cutoff = (((min_cutoff - max_cutoff) * (1 - ((self.initial_depth -
                                                          self.depth) / self.initial_depth))) + max_cutoff)**1.2
        width = int(len(children) * cutoff)
        # print(
        # f'Width: {width} vs {len(children)} Depth: {self.depth} Cutoff: {cutoff}')
        return children[:width]

    def run(self) -> Tuple[int, Union[Space, Tuple[Space, Space]], Direction]:
        if self.depth == 0 or utils.game_is_over(self.game.get_score()):
            return self._end((self._heuristic(self.game), None, None))

        children = self._create_children()
        succ_node = children.pop(0)
        score = self.__class__(
            succ_node[0], self.player, self.depth - 1, self.alpha, self.beta, not self.is_maximizer, succ_node[1], succ_node[2], func=self.func, initial_depth=self.initial_depth).run()
        if score[0] > self.beta:
            return self._end((self.beta, self.marbles, self.direction))
        if score[0] > self.alpha:
            self.alpha = score[0]

        with Pool() as pool:
            children = pool.starmap(run, product(children, [self]))

        for result in children:
            if (result[0] > self.beta):
                return self._end((self.beta, self.marbles, self.direction))
            if (result[0] > self.alpha):
                score = result
                self.alpha = result[0]

        return self._end((self.alpha, score[1], score[2]))


class AlphaBetaPlayer(AbstractPlayer):
    '''
    '''

    def __init__(self, *args, depth=2, verbose=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.verbose = verbose
        self.depth = depth

    def __str__(self):
        return f'AlphaBetaPlayer depth: {self.depth} algo: {str(self.get_algorithm())}'

    def get_algorithm(self):
        return AlphaBetaSimple

    def turn(self, game: Game, moves_history: List[Tuple[Union[Space, Tuple[Space, Space]], Direction]]) -> Tuple[Union[Space, Tuple[Space, Space]], Direction]:
        global nodes
        nodes = 0

        result = self.get_algorithm()(
            game, game.turn.value, func=count_nodes, depth=self.depth, initial_depth=self.depth).run()
        if self.verbose:
            print(f'Heuristic: {result[0]}')
            print(f'Nodes visited: {nodes}')

        return [result[1], result[2]]


class PVSPlayer(AlphaBetaPlayer):
    @ property
    def depth(self):
        return 5

    def get_algorithm(self):
        return PVS


class PVSPlayerShallow(PVSPlayer):
    @ property
    def depth(self):
        return 3


class AlphaBetaPlayerFast(AlphaBetaPlayer):
    def get_algorithm(self):
        return AlphaBetaAdvancedFast


@ dataclass
class Move:
    move: Tuple[Union[Space, Tuple[Space, Space]], Direction]
    value: int
    next_state: Game


class MctsNode:
    def __init__(self, game: Game, parent: MctsNode, marbles: Union[Space, Tuple[Space, Space]] = None, direction: Direction = None):
        self.game = game
        self.parent = parent
        self.marbles = marbles
        self.direction = direction
        self.black_stats = 0
        self.white_stats = 0
        self.no_games = 0
        self.moves = self.create_ordered_moves()
        self.children = []
        self.uct = float('inf')

    @ property
    def stats(self) -> (int, int):
        return (self.black_stats / self.no_games if self.no_games != 0 else 0, self.white_stats / self.no_games if self.no_games != 0 else 0)

    def create_ordered_moves(self) -> List[Move]:
        moves = []
        for move in self.game.generate_legal_moves():
            next_state = copy.deepcopy(self.game)
            next_state.move(*move)
            next_state.switch_player()
            value = Heuristics.evaluate_move(Player.BLACK.value,
                                             self.game, next_state, *move)
            moves.append(Move(
                move=move,
                value=value,
                next_state=next_state,
            ))
        reverse = False if self.game.turn == Player.BLACK else True
        moves.sort(key=lambda x: x.value, reverse=reverse)
        return moves

    def update_uct(self):
        mean = self.black_stats / self.no_games
        no_games_parent = math.log(
            self.parent.no_games + 1) if self.parent else 0
        C = math.sqrt(2)
        self.uct = mean + C * math.sqrt(no_games_parent / self.no_games)

    def append_child(self, child: MctsNode):
        self.children.append(child)

    def update_stats(self, new: Tuple[int, int]):
        self.no_games += 1
        self.black_stats += new[0]
        self.white_stats += new[1]
        self.update_uct()

    def print(self, level=0):
        print(f'level: {level} score: {self.uct}')
        for child in self.children:
            child.print(level+1)


class MonteCarloSearch(Algorithm):
    def __init__(self, game: Game, max_time=5, max_plies=200):
        self.game = game
        self.max_time = max_time
        self.max_plies = max_plies
        self.player = 0 if self.game.turn == Player.BLACK else 1
        self.not_player = 1 if self.game.turn == Player.BLACK else 0
        self.counter = 0
        self.root = MctsNode(self.game, None)

    def initialize(self):
        for move in self.root.moves:
            child = MctsNode(move.next_state, self.root, *move.move)
            self.root.append_child(child)
            self.root.moves.pop()

    def run(self) -> Tuple[Union[Space, Tuple[Space, Space]], Direction]:
        child_count = 0
        sim_count = 0
        start_time = time.time()

        # inititalize tree
        self.initialize()

        # exhaust time resources
        while time.time() - start_time < self.max_time:
            leaf = self.traverse(self.root)
            simulation_result = self.playout(leaf)
            self.backpropagate(leaf, simulation_result)
            child_count += 1
            sim_count += 1
        print(f'child_count: {self.counter} sim_count: {sim_count}')
        # self.root.print()
        return self.choose_best(self.root)

    def choose_best(self, root: MctsNode) -> Tuple[Union[Space, Tuple[Space, Space]], Direction]:
        best = None
        for child in root.children:
            if best is None:
                best = child
            else:
                if best.stats[self.player] < child.stats[self.player]:
                    best = child
        return (best.marbles, best.direction)

    def backpropagate(self, node: MctsNode, result: Tuple[int, int]):
        while (node != None):
            node.update_stats(result)
            node = node.parent

    def playout_policy(self, state: game):
        return state.generate_random_move()

    # function for node traversal
    def traverse(self, node):
        while (True):
            try:
                node = node.children[self.counter]
                self.counter += 1
                break
            except IndexError:
                self.counter = 0
        return node

    def utility(self, state: Game):
        old_score = self.game.get_score()
        new_score = state.get_score()
        marbles_lost = old_score[self.player] - new_score[self.player]
        marbles_won = old_score[self.not_player] - new_score[self.not_player]
        if marbles_won > marbles_lost:
            return (1, 0)
        elif marbles_won < marbles_lost:
            return (0, 1)
        else:
            return (0, 0)

    def playout(self, node: MctsNode):
        plies = 0
        state = copy.deepcopy(node.game)
        while (plies < self.max_plies and not utils.game_is_over(state.get_score())):
            move = self.playout_policy(state)
            state.move(*move)
            state.switch_player()
            plies += 1
        return self.utility(state)


class MonteCarloSearchImproved(MonteCarloSearch):
    def choose_best(self, root: MctsNode) -> Tuple[Union[Space, Tuple[Space, Space]], Direction]:
        best = None
        for child in root.children:
            if best is None:
                best = child
            else:
                if best.no_games < child.no_games:
                    best = child
        return (best.marbles, best.direction)

    def best_uct(self, children):
        best = None
        for child in children:
            if best is None or child.uct > best.uct:
                best = child
        return best

    def initialize(self):
        self.expand(self.root, breadth=20)

    def expand(self, node, breadth=1):
        for _ in range(breadth):
            move = node.moves.pop()
            new_node = MctsNode(move.next_state, node, *move.move)
            node.append_child(new_node)
        return new_node

    def traverse(self, node):
        while (node.children):
            node = self.best_uct(node.children)
        return self.expand(node)


class MonteCarloPlayer(AbstractPlayer):
    def __init__(self, *args, max_time=20, verbose=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_time = max_time

    def __str__(self):
        return f'MonteCarlo player max_time: {self.max_time} algo: {str(self.get_algorithm())}'

    def get_algorithm(self):
        return MonteCarloSearch

    def turn(self, game: Game, moves_history: List[Tuple[Union[Space, Tuple[Space, Space]], Direction]]) -> Tuple[Union[Space, Tuple[Space, Space]], Direction]:
        result = self.get_algorithm()(game, max_time=self.max_time).run()
        return result


class MonteCarloPlayerImproved(MonteCarloPlayer):
    def get_algorithm(self):
        return MonteCarloSearchImproved
