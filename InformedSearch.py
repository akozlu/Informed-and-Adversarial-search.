
############################################################
# Imports
############################################################

from itertools import *
# import pdb
import random
import copy
from collections import deque
import Queue
import math

############################################################
# Section 1: Tile Puzzle
############################################################

#creates the Tile Puzzle object 
def create_tile_puzzle(rows, cols):
    # pdb.set_trace()
    counter = count(1)
    r = rows - 1
    c = cols - 1
    return TilePuzzle([[next(counter) if ((row * col is not r * c)) else 0 for col in xrange(cols)] for row in xrange(rows)])

class TilePuzzle(object):

    
    def __init__(self, board):

        self.board = board
        self.rows = len(board)
        self.cols = len(board[0])

        empty = [[row, col] for col in range(self.cols) for row in range(
            self.rows) if (self.board[row][col] == 0)]

        # a variable to know where the empty tile is at all times
        self.empty = list(chain.from_iterable(empty))

    def get_board(self):
        return (self.board)

    def perform_move(self, direction):

        if direction == "up":
            if self.empty[0] > 0:

               # print("We are performing move UP")
                new_row = self.empty[0] - 1
                new_col = self.empty[1]
                self.apply_move(new_row, new_col)
                return True

        elif direction == "down":
            if self.empty[0] < self.rows - 1:
               # print("We are performing move DOWN")
                new_row = self.empty[0] + 1
                new_col = self.empty[1]
                self.apply_move(new_row, new_col)
                return True

        elif direction == "left":

            if self.empty[1] > 0:
                # print("We are performing move LEFT")

                new_row = self.empty[0]
                new_col = self.empty[1] - 1
                self.apply_move(new_row, new_col)

                return True

        elif direction == "right":

            if self.empty[1] < self.cols - 1:
                # print("We are performing move RIGHT")
                new_row = self.empty[0]
                new_col = self.empty[1] + 1
                self.apply_move(new_row, new_col)
                return True

        return False

    def __str__(self):

        # Function is used testing. Visualliy always knowing where the empty
        # tile is useful for debugging.

        return ", ".join(str(x) for x in self.empty)

    def apply_move(self, row, col):

        number = self.board[row][col]
        self.board[self.empty[0]][self.empty[1]] = number
        self.board[row][col] = 0
        self.empty = [row, col]

    def scramble(self, num_moves):

        for i in range(num_moves):
            self.perform_move(random.choice(['up', 'down', 'left', 'right']))

    def is_solved(self):

        counter = count(1)

        # creates a single list
        iterable_board = chain.from_iterable(self.board)

        # if every element of the list except the last is in increasing order
        # return true
        first_condition = all(next(counter) == i for i in islice(
            iterable_board, ((self.rows * self.cols) - 1)))

        # last element must be zero
        second_condition = (self.board[self.rows - 1][self.cols - 1] == 0)

        return (first_condition and second_condition)

    def copy(self):

        return copy.deepcopy(self)

    def successors(self):

        for direction in ['up', 'down', 'left', 'right']:
            new_board = self.copy()
            validBoard = new_board.perform_move(direction)
            if validBoard:
                yield direction, new_board

    def toTuple(self):
        return tuple(tuple(row) for row in self.get_board())

    # Required
    def find_solutions_iddfs(self):

        # pdb.set_trace()

        depth = 0
        found_solution = False

        while True:

            if found_solution:
                break

            result = self.iddfs_helper(depth)

            if result:

                for solution in result:
                    yield solution
                    found_solution = True
            else:
                depth = depth + 1

    def iddfs_helper(self, depth):

        solutions = []

        # We have a stack of tuples representing (moves,board)
        # Moves is a list of moves that was made in order to reach the board
        # state. I used it to backtrack all optimal solutions.
        stack = []
        # we will keep the move list of every board state as a list
        initial_move_list = []
        number_of_nodes_visited = 0

        stack.append((initial_move_list, self))

        # dictionary to map board tuples to their depths. It is used so that we
        # don't expand board states seen at shallower depths.

        board_depth_dictionary = {}

        # The starting configuration will always be at depth 0.

        board_depth_dictionary[self.toTuple()] = 0

        while stack:

            currBoard = stack.pop()

            board = currBoard[1]

            moves = currBoard[0]

            if(board.is_solved()):

                solutions.append(moves)

            # length of the move list is our current depth

            if(len(moves) < depth):

                for move, newBoard in board.successors():

                    # If we have already seen this board configuration at a
                    # SHALLOWER Depth, we shouldn't expand it. For depth = 6,
                    # this cuts our search in half!

                    if (newBoard.toTuple() in board_depth_dictionary and board_depth_dictionary[newBoard.toTuple()] < len(moves) + 1):

                        continue

                    new_move_list = moves[:]
                    new_move_list.append(move)
                    stack.append((new_move_list, newBoard))
                    new_move_list = []
                    number_of_nodes_visited = number_of_nodes_visited + 1

       # print("NUMBER OF VISITED NODES")
        print(number_of_nodes_visited)
        return solutions

    def ManhattanDistance(self):

        cols = self.cols
        board = list(chain.from_iterable(self.get_board()))

        # we only need to know the number of columns to figure out the
        # Manhattan Distance. The Filter makes sure we dont check the MH for 0.
        return sum(abs((val - 1) // cols - i // cols) + abs((val - 1) % cols - i % cols) for i, val in enumerate(board) if val)

    def find_solution_a_star(self):

        q = Queue.PriorityQueue()

        # Priority Queue with two priority values
        # Function G is used as a second priority to improve efficiency.

        q.put((0, 0, self))
       # number_of_nodes_visited = 0

        explored = set()

        came_from = {}
        came_from[self] = None

        cost_so_far = {}

        cost_so_far[self] = 0

        moves = {}
        moves[self] = None

        while not q.empty():

            tup = q.get()

            board = tup[2]

            explored.add(board.toTuple())

            if board.is_solved():

                end_node = board
                sol = []
                while not came_from[end_node] == None:
                    sol.append((moves[end_node]))
                    end_node = came_from[end_node]
                   # print(number_of_nodes_visited)
                return (list(reversed(sol)))

            else:

                for move, nextBoard in board.successors():

                    if nextBoard.toTuple() not in explored:

                        G = tup[1] + 1

                        H = nextBoard.ManhattanDistance()
                        F = H + G
                        #number_of_nodes_visited = number_of_nodes_visited + 1
                        moves[nextBoard] = move
                        came_from[nextBoard] = board
                        q.put((F, G, nextBoard))

                        
"""
Testing Purposes
b = [[4, 1, 2], [0, 5, 3], [7, 8, 6]]
p = TilePuzzle(b)
print(p.find_solution_a_star())
b = [[1, 2, 3], [4, 0, 5], [6, 7, 8]]
p = TilePuzzle(b)
print(p.find_solution_a_star())

# solutions = p.find_solutions_iddfs()
# print(solutions)
# print(p)


b = [[4, 1, 2], [0, 5, 3], [7, 8, 6]]
p = TilePuzzle(b)
b = [[1, 2, 3], [4, 0, 8], [7, 6, 5]]
p = TilePuzzle(b)
print(list(p.find_solutions_iddfs()))

print(p.is_solved())
p.perform_move("up")
print(p.get_board())
print(p.is_solved())
p.perform_move("down")
print(p.is_solved())

b = [[1, 2, 3], [4, 0, 5], [6, 7, 8]]
p = TilePuzzle(b)
for move, new_p in p.successors():
    print move, new_p.get_board()
p = create_tile_puzzle(3, 3)
for move, new_p in p.successors():
    print move, new_p.get_board()
"""

############################################################
# Section 2: Grid Navigation
############################################################


def isValid(a, max_row_col):

    return (a > 0) and (a < max_row_col)


def find_path(start, goal, scene):

    # This function uses A* SEARCH with couple of optimizations to pick
    # certain nodes with higher chance of being closer to the goal state from
    # PQ

    frontier = Queue.PriorityQueue()

    # Again we use a double priority queue.

    frontier.put((0, 0, start))

    came_from = {}
    cost_so_far = {}

    came_from[start] = None
    cost_so_far[start] = 0

    # If the given start is at a blocked position, return None. Saw on Piazza
    # that we need to cover this case as well.

    if scene[start[0]][start[1]] or scene[goal[0]][goal[1]]:
        return None

    while not frontier.empty():

        first = frontier.get()

        current = first[2]

        # If we reach goal state, backtrack the solution and return it
        if current == goal:

            position = current
            sol = [current]
            while not came_from[position] == None:
                sol.append((came_from[position]))
                position = came_from[position]
            return (list(reversed(sol)))

        for nextCell in get_successors(current[0], current[1], scene):

            new_cost = cost_so_far[current] + 1

            # If we haven't seen this cell before or the cost of was more than
            # its previous cost of reaching it.  We update its costs and use
            # this new cost to calculate F and G. Then we can add it to PQ with
            # priorities F and G.

            if nextCell not in cost_so_far:

                cost_so_far[nextCell] = new_cost
                F = new_cost + euclid_distance(nextCell, goal)
                print(F)
                frontier.put((F, new_cost, nextCell))
                came_from[nextCell] = current
    return None


def euclid_distance(current, goal):
    return math.sqrt((goal[0] - current[0])**2 + (goal[1] - current[1])**2)


def get_successors(row, col, scene):

    # This function generates all possible directions out of 8 possible
    # directions

    rows = len(scene)
    cols = len(scene[0])

    # down, up, left, right

    if row > 0 and not scene[row - 1][col]:

        yield((row - 1, col))

    if row < rows - 1 and not scene[row + 1][col]:

        yield((row + 1, col))

    if col > 0 and not scene[row][col - 1]:

        yield((row, col - 1))

    if col < cols - 1 and not scene[row][col + 1]:

        yield((row, col + 1))

    # NE

    if row > 0 and col > 0 and not scene[row - 1][col - 1]:

        yield((row - 1, col - 1))

    # NW

    if row > 0 and col < cols - 1 and not scene[row - 1][col + 1]:

        yield((row - 1, col + 1))

    # SE
    if row < rows - 1 and col > 0 and not scene[row + 1][col - 1]:

        yield((row + 1, col - 1))

    # SW
    if row < rows - 1 and col < cols - 1 and not scene[row + 1][col + 1]:

        yield((row + 1, col + 1))

"""
Tests for debugging

scene = [[False, False, False], [False, True, False], [False, False, False]]
print(len(list(get_successors(1, 1, scene))))

scene2 = [[False, True, False], [False, True, False], [True, False, False]]

print(list(get_successors(1, 1, scene)))
print(list(get_successors(2, 1, scene2)))

print(find_path((0, 0), (2, 1), scene))
"""
############################################################
# Section 3: Linear Disk Movement, Revisited
############################################################


class LinearDiskSolver(object):

    def __init__(self, disks, length, n):

        self.board = list(disks)
        self.length = length
        self.n = n

    def get_board(self):

        return self.board

    def __str__(self):

        # Function is used for testing purposes.
        return ", ".join(str(x) for x in self.get_board())

    def copy(self):

        return copy.deepcopy(self)

    def is_solved(self):

        for i in xrange(self.length - self.n):

            if self.board[i] is not None:
                return False

        for i in xrange(self.length - self.n, self.length):

            if self.board[i] is not self.length - i:
                return False
        return True

    def apply_move(self, i, steps):

        # Helper function that modifies the board for all of the different
        # steps
        if (i + steps) < 0 or (i + steps) >= self.length:
            return
        self.board[i + steps] = self.board[i]
        self.board[i] = None

    def successors(self):

        # Similar approach taken from the LightsOutPuzzle problem
        board = self.get_board()

        length = self.length

        for i, cell in enumerate(board):

            if cell is not None:

                if i + 1 < length and board[i + 1] is None:

                    newBoard = self.copy()
                    newBoard.apply_move(i, 1)
                    yield(tuple((i, i + 1)), newBoard)

                if i + 2 < length and board[i + 2] is None and board[i + 1] is not None:

                    newBoard = self.copy()
                    newBoard.apply_move(i, 2)
                    yield(tuple((i, i + 2)), newBoard)

                if i - 1 >= 0 and board[i - 1] is None:

                    newBoard = self.copy()
                    newBoard.apply_move(i, -1)
                    yield(tuple((i, i - 1)), newBoard)

                if i - 2 >= 0 and board[i - 2] is None and board[i - 1] is not None:

                    newBoard = self.copy()
                    newBoard.apply_move(i, -2)
                    yield(tuple((i, i - 2)), newBoard)

    def solution_for_distinct(self):

        frontier = Queue.PriorityQueue()

    # Again we use a double priority queue. Solution found similar to the Grid
    # Navigation Problem.

        frontier.put((0, 0, self))

        came_from = {}
        cost_so_far = {}
        moves = {}

        came_from[self] = None
        cost_so_far[self] = 0
        moves[self] = None

        solution = []

        while not frontier.empty():

            first = frontier.get()

            board = first[2]

            if board.is_solved():

                position = board
                while not came_from[position] == None:
                    solution.append(moves[position])
                    position = came_from[position]

                return list(reversed(solution))

            for move, nextBoard in board.successors():

                new_cost = cost_so_far[board] + 1

                if nextBoard not in cost_so_far or new_cost < cost_so_far[nextBoard]:

                    cost_so_far[nextBoard] = new_cost
                    F = new_cost + nextBoard.get_heuristic()
                    frontier.put((F, new_cost, nextBoard))
                    came_from[nextBoard] = board
                    moves[nextBoard] = move

        return None

    def get_heuristic(self):

        # An admissible heuristic: Calculate how far away each disk is away
        # from their final position.  For example, how far away is first disk from
        # cell n-1, how far away is the second disk (originally on cell 1) to
        # cell n-2? Distance is defined as number of single moves for a disk to reach the target cell.
        # We loop over the board and add these distances together. This heuristic is
        # admissible, it never overestimates the number of moves required to reach a goal state.
        # Although I will calculate if it will yield SIGNIFICANT improvements.

        H = 0
        board = self.get_board()
        length = self.length

        for i, cell in enumerate(board):

            if cell is not None:

                # if cell is not in destination position
                if cell != board[-1 * cell]:

                    # Required destination for cell
                    final_posiiton = length - cell

                    # calculate distance between current position and
                    # destination
                    H = H + abs(final_posiiton - i)
        return H


def solve_distinct_disks(length, n):
    disks = tuple(i + 1 if i < n else None for i in xrange(length))

    return LinearDiskSolver(disks, length, n).solution_for_distinct()


def create_disk_lists(disks, length, n):
    # Function was for testing purposes.
    return LinearDiskSolver(disks, length, n)


def create_disk_list(length, n):
    # Function was for testing purposes.
    disks = tuple(i + 1 if i < n else None for i in xrange(length))

    return LinearDiskSolver(disks, length, n)

"""
p = create_disk_list(5, 3)

print(p.get_heuristic())
b = [None, None, 3, 2, 1]
p1 = create_disk_lists(b, 5, 3)
print(p1.get_heuristic())
b = [1, None, 2, None, None]
p1 = create_disk_lists(b, 5, 2)

print(solve_distinct_disks(5, 2))
print(solve_distinct_disks(4, 2))


print(len(solve_distinct_disks(16, 8)))
print(len(solve_distinct_disks(18, 6)))
"""

############################################################
# Section 4: Dominoes Game
############################################################


def create_dominoes_game(rows, cols):
    pass


class DominoesGame(object):

    # Required
    def __init__(self, board):
        pass

    def get_board(self):
        pass

    def reset(self):
        pass

    def is_legal_move(self, row, col, vertical):
        pass

    def legal_moves(self, vertical):
        pass

    def perform_move(self, row, col, vertical):
        pass

    def game_over(self, vertical):
        pass

    def copy(self):
        pass

    def successors(self, vertical):
        pass

    def get_random_move(self, vertical):
        pass

    # Required
    def get_best_move(self, vertical, limit):
        pass

