from env.cmdp import CMDP
import numpy as np
from typing import List, Tuple, Dict, Optional, Any, Union, Callable

Point = np.ndarray  # type for state and action coordinates, np.array([int, int])

class Gridworld(CMDP):
    """
    Gridworld MDP.

    additional Attributes:
        actions: List of available actions as numpy arrays [a_right, a_up].
        grid_height: Integer for grid height.
        grid_width: Integer for grid width.
        noise: Chance of moving randomly.
    """

    def __init__(self, grid_width: int, grid_height: int, noise: float, gamma: float,
                 nu0: Optional[np.ndarray] = None, r: Optional[np.ndarray] = None,
                 constraints: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> None:

        # gridworld specific attributes
        self.actions: List[Point] = [np.array([1, 0]), np.array([0, 1]), np.array([-1, 0]), np.array([0, -1])]
        self.grid_height: int = grid_height
        self.grid_width: int = grid_width
        self.noise = noise

        # general CMDP attributes
        self.n = grid_width * grid_height
        self.m = len(self.actions)
        print("construct transition matrix")
        self.P: np.ndarray = np.array(
            [[[self._transition_dynamics(s, a, s_next)
               for s_next in range(self.n)]
                for a in range(self.m)]
                 for s in range(self.n)])
        print("transition matrix constructed")
        super().__init__(self.n, self.m, gamma, P=self.P, nu0=nu0, r=r, constraints=constraints)

    def _transition_dynamics(self, s: int, a: int, s_next: int) -> float:
        """
        Get the probability of transitioning from state s to state s_next given
        action a.

        :param s: State int.
        :param a: Action int.
        :param s_next: State int.
        :return: P(s_next | s, a)
        """

        s_next = self.int2point(s_next)
        s = self.int2point(s)
        a = self.actions[a]

        if not self.neighbouring(s_next, s):
            return 0.0

        # Is s_next the intended state to move to?
        if (s + a == s_next).all():
            return 1 - self.noise + self.noise / self.m

        # If these are not the same point, then we can move there by noise.
        if not (s == s_next).all():
            return self.noise / self.m

        # If these are the same point, we can only move here by either moving
        # off the grid or being blown off the grid. Are we on a corner or not?
        if (s == np.array([0, 0])).all() or (s == np.array([self.grid_width - 1, self.grid_height - 1])).all() \
                or (s == np.array([0, self.grid_height - 1])).all() or (s == np.array([self.grid_width - 1, 0])).all():
            # Corner.
            # Can move off the edge in two directions.
            # Did we intend to move off the grid?
            if not (0 <= s + a).all() and (s + a < np.array([self.grid_width, self.grid_height])).all():
                # We intended to move off the grid, so we have the regular
                # success chance of staying here plus an extra chance of blowing
                # onto the *other* off-grid square.
                return 1 - self.noise + 2 * self.noise / self.m
            else:
                # We can blow off the grid in either direction only by noise.
                return 2 * self.noise / self.m
        else:
            # Not a corner. Is it an edge?
            if (s[0] not in {0, self.grid_width - 1} and
                    s[1] not in {0, self.grid_height - 1}):
                # Not an edge.
                return 0.0

            # Edge.
            # Can only move off the edge in one direction.
            # Did we intend to move off the grid?
            if not (0 <= s + a).all() and (s + a < np.array([self.grid_width, self.grid_height])).all():
                # We intended to move off the grid, so we have the regular
                # success chance of staying here.
                return 1 - self.noise + self.noise / self.m
            else:
                # We can blow off the grid only by noise.
                return self.noise / self.m

    def _transition_dynamics_old(self, j: Point, k: Point, i: Point) -> float:
        """
        Get the probability of transitioning from state j to state i given
        action k.

        :param i: State int.
        :param j: State int.
        :param k: Action int.
        :return: p(s_i | s_j, a_k)
        """

        xi, yi = self.int2point(i)
        xj, yj = self.int2point(j)
        xk, yk = self.actions[k]

        if not self.neighbouring((xi, yi), (xj, yj)):
            return 0.0

        # Is i the intended state to move to?
        if (xj + xk, yj + yk) == (xi, yi):
            return 1 - self.noise + self.noise / self.m

        # If these are not the same point, then we can move there by noise.
        if (xj, yj) != (xi, yi):
            return self.noise / self.m

        # If these are the same point, we can only move here by either moving
        # off the grid or being blown off the grid. Are we on a corner or not?
        if (xj, yj) in {(0, 0), (self.grid_width - 1, self.grid_height - 1),
                        (0, self.grid_height - 1), (self.grid_width - 1, 0)}:
            # Corner.
            # Can move off the edge in two directions.
            # Did we intend to move off the grid?
            if not (0 <= xk + xj < self.grid_width and
                    0 <= yk + yj < self.grid_height):
                # We intended to move off the grid, so we have the regular
                # success chance of staying here plus an extra chance of blowing
                # onto the *other* off-grid square.
                return 1 - self.noise + 2 * self.noise / self.m
            else:
                # We can blow off the grid in either direction only by noise.
                return 2 * self.noise / self.m
        else:
            # Not a corner. Is it an edge?
            if (xj not in {0, self.grid_width - 1} and
                    yj not in {0, self.grid_height - 1}):
                # Not an edge.
                return 0.0

            # Edge.
            # Can only move off the edge in one direction.
            # Did we intend to move off the grid?
            if not (0 <= xk + xj < self.grid_width and
                    0 <= yk + yj < self.grid_height):
                # We intended to move off the grid, so we have the regular
                # success chance of staying here.
                return 1 - self.noise + self.noise / self.m
            else:
                # We can blow off the grid only by noise.
                return self.noise / self.m

    # basic functionality
    def neighbouring(self, i: Point, k: Point) -> bool:
        """
        Get whether two points neighbour each other. Also returns true if they
        are the same point.

        :param i: (x, y) int tuple.
        :param k: (x, y) int tuple.
        :return: Boolean.
        """

        return abs(i[0] - k[0]) + abs(i[1] - k[1]) <= 1

    def int2point(self, i: int) -> Point:
        """
        Convert a state int into the corresponding coordinate.

        :param i: State int.
        :return: (x, y) int tuple.
        """

        return np.array([i % self.grid_width, i // self.grid_width])


    def action2int(self, a: Point) -> int:
        """
        Convert an action such as [1,0] to an action integer.

        :param a: Action.
        :return: Corresponding integer.
        """

        return self.actions.index(a)