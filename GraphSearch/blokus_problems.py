import math
import random
from typing import List, Tuple

from board import Board, Move
from search import SearchProblem, depth_first_search
import util
from pieces import PieceList, Piece

EMPTY_BOARD_CELL = -1
PLAYER_INDEX = 0
INFINITY = float('inf')


class BlokusFillProblem(SearchProblem):
    """
    A one-player Blokus game as a search problem.
    This problem is implemented for you. You should NOT change it!
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0)):
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)
        self.expanded = 0

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        """
        state: Search state
        Returns True if and only if the state is a valid goal state
        """
        return not any(state.pieces[0])

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, 1) for move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        return len(actions)


#####################################################
# This portion is incomplete.  Time to write code!  #
#####################################################


class BlokusCornersProblem(SearchProblem):
    _NUM_PLAYERS = 1

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0)):
        self.expanded = 0
        self.starting_point = starting_point
        "*** YOUR CODE HERE ***"
        self.board = Board(board_w, board_h, BlokusCornersProblem._NUM_PLAYERS,
                           piece_list, starting_point)

        # top right point, bottom right point, top left point, bottom left point
        self.corners = [(0, 0), (board_h - 1, 0), (0, board_w - 1), (board_h - 1, board_w - 1)]
        self.corners.remove(starting_point)

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state: Board) -> bool:
        """
        Returns True if all three target corners of the board are occupied
        """
        for corner in self.corners:
            if state.get_position(corner[1], corner[0]) == EMPTY_BOARD_CELL:
                return False
        return True

    def get_successors(self, state: Board):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, move.piece.get_num_tiles()) for move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions: List[Move]):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        "*** YOUR CODE HERE ***"
        cost = 0
        for move in actions:
            cost += move.piece.get_num_tiles()
        return cost


class BlokusCoverProblem(SearchProblem):
    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0), targets=[(0, 0)]):
        self.starting_point = starting_point
        self.targets = targets.copy()
        self.expanded = 0
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)
        if starting_point in self.targets:
            self.targets.remove(starting_point)

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state: Board) -> bool:
        for target in self.targets:
            if state.get_position(target[1], target[0]) == EMPTY_BOARD_CELL:
                return False
        return True

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, move.piece.get_num_tiles()) for move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        "*** YOUR CODE HERE ***"
        cost = 0
        for move in actions:
            cost += move.piece.get_num_tiles()
        return cost


def set_calculator(problem, targets):
    if BlokusHeuristic.heuristic_calculator is None or problem != BlokusHeuristic.heuristic_calculator.problem:
        BlokusHeuristic.heuristic_calculator = BlokusHeuristic(problem, targets)


class BlokusHeuristic:
    """
    A heuristic for the Blokus game that estimates the cost of reaching target positions on the board.

    Attributes:
        heuristic_calculator (BlokusHeuristic): A global instance of the BlokusSimulator.
    """
    heuristic_calculator: "BlokusHeuristic" = None

    def __init__(self, problem, targets, ):
        """
        Initializes the BlokusSimulator with a given problem, simulation type, and target positions.

        Args:
            problem: The initial problem state for the simulation.
            simulation_type (str): The type of simulation ('cover' or 'corner').
            targets (list): The list of target positions to reach on the board.
        """
        self.problem = problem
        self.targets = targets.copy()

        self.positive_targets = [(x, y) for (x, y) in targets if x >= problem.starting_point[0]]
        self.positive_targets.sort(key=lambda target: self.distance(target, problem.starting_point))
        self.negative_targets = [(x, y) for (x, y) in targets if x < problem.starting_point[0]]
        self.negative_targets.sort(key=lambda target: self.distance(target, problem.starting_point))

        self.cost_estimation = dict()  # state : [estimation, depth]
        self.explored = 0
        self.pieces = self.sorted_pieces(self.problem.get_start_state().piece_list)


    def distance(self, pos1, pos2):
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


    def sorted_pieces(self, piece_list: PieceList):
        """
        Sorts the pieces based on their diagonal advancement and the number of tiles they cover.

        Args:
            piece_list (PieceList): The list of pieces to be sorted.

        Returns:
            list: A sorted list of tuples containing piece information.
        """
        x_advance = lambda p: len(set(p.x))
        y_advance = lambda p: len(set(p.y))
        diagonal_advance = lambda p: math.sqrt(x_advance(p) ** 2 + y_advance(p) ** 2)
        pieces = [(p.get_num_tiles(), diagonal_advance(p), index)
                  for index, p in enumerate(piece_list)]
        pieces.sort(key=lambda tup: tup[0] / tup[1])
        for piece in piece_list:
            print(piece.x, piece.y)
        print(pieces)
        return pieces

    def get_cost_estimation(self, s):
        """
        Retrieves or calculates the cost estimation for a given state.

        Args:
            s: The state for which the cost estimation is to be retrieved or calculated.

        Returns:
            The cost estimation for the given state.
        """
        if s not in self.cost_estimation:
            self.cost_estimation[s] = self.block_distance(self.euclidian_distance(s), s.pieces)
        return self.cost_estimation[s]

    def block_distance(self, euc_distance, legal_pieces):
        """
        Calculates the block distance based on Euclidean distance and the legality of pieces.

        Args:
            euc_distance (float): The Euclidean distance to the target.
            legal_pieces (list): The list of legal pieces.

        Returns:
            The total cost based on the block distance.
        """
        if euc_distance == INFINITY:
            return INFINITY
        minimal_distance = euc_distance
        distance = 0
        total_cost = 0
        for cost, steps, index in self.pieces:
            if not legal_pieces[PLAYER_INDEX, index]:
                continue
            if distance + steps >= minimal_distance:
                distance_left = (minimal_distance - distance)
                frac = distance_left / steps
                total_cost += cost * frac
                distance = minimal_distance
                break
            distance += steps
            total_cost += cost
        if distance < minimal_distance:
            return INFINITY
        else:
            return total_cost # / len(self.targets)

    def minimize_path_cost(self, path: List[Move]):
        """
        Minimizes the path cost by updating the cost estimations for each move in the path.

        Args:
            path (List[Move]): The path to be minimized.
        """
        board: Board = self.problem.get_start_state()
        self.cost_estimation[board] = 0 if len(path) != 0 else INFINITY
        for move in path:
            board = board.do_move(PLAYER_INDEX, move)
            self.cost_estimation[board] = 0

    def in_board(self, state, pos):
        """
        Checks if a position is within the board boundaries.

        Args:
            state: The current state of the board.
            pos (tuple): The position to check.

        Returns:
            bool: True if the position is within the board boundaries, False otherwise.
        """
        return state.board_w > pos[0] >= 0 and state.board_h > pos[1] >= 0

    def target_blocked(self, state: Board):
        """
        Checks if the target positions are blocked by the player's pieces.

        Args:
            state (Board): The current state of the board.

        Returns:
            bool: True if any target position is blocked, False otherwise.
        """
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for target in self.targets:
            if state.get_position(target[1], target[0]) == PLAYER_INDEX:
                continue
            for dir in directions:
                side = self.add(target, dir)
                x, y = side
                if self.in_board(state, self.add(target, dir)) and state.get_position(y, x) == PLAYER_INDEX:
                    return True
        return False

    def add(self, u, v):
        """
        Returns the addition of two vectors.
        """
        return (u[0] + v[0], u[1] + v[1])

    def satisfied_target(self, target, state):
        x, y = target
        return state.get_position(y, x) == PLAYER_INDEX

    def get_targets(self, state):
        targets = []
        for target_lst in [self.positive_targets, self.negative_targets]:
            for t in target_lst:
                if not self.satisfied_target(t, state):
                    targets.append(t)
                    break
        return targets

    def euclidian_distance(self, state: Board):
        if self.target_blocked(state):
            return INFINITY
        distance = [0 if state.get_position(y, x) == PLAYER_INDEX else INFINITY for x, y in self.targets]
        legal = []
        for x in range(state.board_w):
            for y in range(state.board_h):
                if state.check_tile_attached(PLAYER_INDEX, x, y) \
                        and state.check_tile_legal(PLAYER_INDEX, x, y):
                    legal.append((x, y))
        for x, y in legal:
            for i in range(3):
                corner_x, corner_y = self.targets[i]
                dis = math.sqrt((corner_x - x) ** 2 + (corner_y - y) ** 2)
                if dis < distance[i]:
                    distance[i] = dis
        return sum(distance)

    # def euclidian_distance(self, state: Board):
    #     """
    #     Calculates the Euclidean distance to the target positions.
    #
    #     Args:
    #         state (Board): The current state of the board.
    #
    #     Returns:
    #         float: The sum of Euclidean distances to the target positions.
    #     """
    #     if self.target_blocked(state):
    #         return INFINITY
    #     distance = [INFINITY if not self.satisfied_target(t, state) else 0 for t in self.targets]
    #     legal = []
    #     for x in range(state.board_w):
    #         for y in range(state.board_h):
    #             if state.check_tile_attached(PLAYER_INDEX, x, y) \
    #                     and state.check_tile_legal(PLAYER_INDEX, x, y):
    #                 legal.append((x, y))
    #     for tile in legal:
    #         for i in range(len(self.targets)):
    #             corner = self.targets[i]
    #             dis = self.distance(tile, corner)
    #             if dis < distance[i]:
    #                 distance[i] = dis
    #     return sum(distance)


def blokes_generic_heuristic(state, problem, targets):
    set_calculator(problem, targets)
    return BlokusHeuristic.heuristic_calculator.get_cost_estimation(state)


def blokus_corners_heuristic(state: Board, problem: BlokusCornersProblem):
    """
    Your heuristic for the BlokusCornersProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come up
    with an admissible heuristic; almost all admissible heuristics will be consistent
    as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the other hand,
    inadmissible or inconsistent heuristics may find optimal solutions, so be careful.
    """
    return blokes_generic_heuristic(state, problem, problem.corners)


def blokus_cover_heuristic(state, problem: BlokusCoverProblem):
    """
    Heuristic function for the Blokus Cover problem. It estimates the cost of covering target positions on the board.

    This function uses a global instance of the BlokusHeuristic to avoid reinitializing it for the same problem.
    If the global simulator is not initialized or is initialized with a different problem, it reinitializes the simulator.

    Args:
        state: The current state of the Blokus board.
        problem (BlokusCoverProblem): The Blokus cover problem instance containing the targets to be covered.

    Returns:
        The estimated cost of covering the targets from the current state.
    """
    return blokes_generic_heuristic(state, problem, problem.targets)


#   Tests!!!
############################################################
from optparse import OptionParser

usage_str = """
USAGE:      python game.py <options>
EXAMPLES:  (1) python game.py
              - starts a game between 4 random agents
           (2) python game.py -p tiny_set.txt -s 4 7
           OR  python game.py -s 14 14 -f ucs -z cover [(1, 1), (5, 9), (9, 6)]
"""
parser = OptionParser(usage_str)

parser.add_option('-p', '--pieces', dest='pieces_file',
                  help='the file to read for the list of pieces',
                  default='valid_pieces.txt')
parser.add_option('-s', '--board-size', dest='size',
                  type='int', nargs=2, help='the size of the game board.', default=(20, 20))
parser.add_option('-f', '--search-function', dest='search_func',
                  metavar='FUNC', help='search function to use. This option is ignored for sub-optimal search. ',
                  type='choice',
                  choices=['dfs', 'bfs', 'ucs', 'astar'], default='dfs')
parser.add_option('-H', '--heuristic', dest='h_func',
                  help='heuristic function to use for A* search. \
                  This option is ignored for other search functions. ',
                  metavar='FUNC', default=None)
parser.add_option('-z', '--puzzle', dest='puzzle',
                  help='the type of puzzle being solved', type='choice',
                  choices=['fill', 'diagonal', 'corners', 'cover', 'sub-optimal', 'mini-contest'], default=None)
parser.add_option('-x', '--start-point', dest='start', type='int', nargs=2,
                  help='starting point', default=(0, 0))

options, cover_points = parser.parse_args()
piece_list = PieceList(options.pieces_file)
depth_first_search(BlokusFillProblem(20, 20, piece_list))
