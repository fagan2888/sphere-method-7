import unittest
import sphere as sp
import math
import numpy as np

# Here's our "unit".

    # # Constraint class
    # a1 = sp.Constraint([1, 2, 3], 3)
    # #     a1 =  sp.Constraint([1, 1, 1], 3)
    # x = np.array([0, 0, 0])
    # print(a1.directedDistanceFrom(x))
    # print(a1.nearestPoint(x))
    #
    # # construct an LP, and solve it
    # a1 = sp.Constraint([-1, -1, -1], -3)
    # a2 = sp.Constraint([-1, -2, -3], -6)
    # a3 = sp.Constraint([1, 0, 0], 0)
    # a4 = sp.Constraint([0, 1, 0], 0)
    # a5 = sp.Constraint([0, 0, 1], 0)
    # A = sp.Constraints([a1, a2])
    # c = sp.ObjectiveFunction([-1, -2, -1])
    # lp = sp.LinearProgram(A, c)
    # print(lp)
    # lp.setIFS([0.5, 0.5, 0.5])
    # print(lp.ifs)
    # lp.solve()

# Here's our "unit tests".
class Subroutine1Tests(unittest.TestCase):

    def test1(self):
        a = [-3]
        g = [1]
        result = sp.subroutine1(a, g)
        self.assertEqual(result[0], 3)
        self.assertEqual(result[1], float("inf"))

    def test2(self):
        a = [6]
        g = [5]
        result = sp.subroutine1(a, g)
        self.assertEqual(result[0], -1.2)
        self.assertEqual(result[1], float("inf"))

    def test3(self):
        a = [6]
        g = [-5]
        result = sp.subroutine1(a, g)
        self.assertEqual(result[0], float("-inf"))
        self.assertEqual(result[1], 1.2)

    def testN(self):
        a = [-2, -1, 3, 8]
        g = [1, 2, -1, -2]
        result = sp.subroutine1(a, g)
        self.assertEqual(result[0], 2)
        self.assertEqual(result[1], 3)

class ConstraintTests(unittest.TestCase):
    def test_directed_distance(self):
        a1 = sp.Constraint([1, 2, 3], 3)
        x = np.array([0, 0, 0])
        self.assertEqual(a1.directedDistanceFrom(x), - 3 / np.sqrt(14))


    def test1(self):
        a1 = sp.Constraint([1, 2, 3], 3)
        x = np.array([0, 0, 0])
        nearest = a1.project(x)
        # would like this to run with Python 3, but for now must cast ints to float
        self.assertAlmostEqual(nearest[0], 3.0 / 14)
        self.assertAlmostEqual(nearest[1], 6.0/ 14)
        self.assertAlmostEqual(nearest[2], 9.0 / 14)

class TouchingConstraintsTests(unittest.TestCase):
    def test_average_direction(self):
        a1 = sp.Constraint([-1, -1, -1], -3)
        a2 = sp.Constraint([-1, -2, -3], -6)
        A = sp.Constraints([a1, a2])
        c = sp.ObjectiveFunction([-1, -2, -1])
        lp = sp.LinearProgram(A, c)
        lp.setIFS([0.5, 0.5, 0.5])
        # delta_x_hat, touching_constraints = A.deltaAndTouchingConstraints(lp.ifs)
        avg_dir= lp.ifs.average_direction_of_touching_constraints()
        self.assertAlmostEqual(avg_dir[0], -1)

class ImplementationDetailTests(unittest.TestCase):
    def test_optimal_delta(self):
        A_input = [[-1.5, -1],
                   [-1.0, -1],
                   [-0.3, -0.5],
                   [-1, 0],
                   [0, -1],
                   [1, 0],
                   [0, 1],
                   ]
        b_input = [-27, -21, -9, -15, -16, 0, 0]
        A = sp.Constraints([sp.Constraint(a, b) for a, b in zip(A_input, b_input)])
        c = sp.ObjectiveFunction([-600, -100])
        lp = sp.LinearProgram(A, c)
        lp.setIFS([1, 3])
        lp.solve()
        self.assertAlmostEqual(lp.imp_detail_alpha, 10.5714285714)
        self.assertAlmostEqual(lp.imp_detail_delta, 1.28571428571)

class NumericalTestToEnsureRunningPython3(unittest.TestCase):
    def test_integerDivide(self):
        self.assertAlmostEqual(0.5, 1/2)

def main():
    unittest.main()

if __name__ == '__main__':
    main()
