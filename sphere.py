'''
Created on Dec 13, 2016

@author: pbenson
'''

import numpy as np
from gurobipy import *

def norm_sqd(x):
    return np.sum(x * x)

def subroutine1(a, g):
    v1 = float("-inf")
    v1_ratios = [ -float(a_t) / g_t for a_t, g_t in zip(a,g) if g_t > 0]
    if v1_ratios:
        v1 = max(v1_ratios)
    v2 = float("inf")
    v2_ratios = [ (-float(a_t) / g_t) for a_t, g_t in zip(a,g) if g_t < 0]
    if v2_ratios:
        v2 = min(v2_ratios)
    if v1 > v2:
        return None
    return (v1, v2)
            
class LinearProgram:
    def __init__(self, constraints, objectiveFunction):
        self.constraints = constraints
        self.objectiveFunction = objectiveFunction

    def __str__(self):
        return 'Linear Program:\n' + str(self.constraints) + '\n' + str(self.objectiveFunction)
        
    def setIFS(self, x):
        self.ifs = FeasibleSolution(x, self.constraints)
        # self.x_bar_hat = self.x_hat - self.ifs.delta * self.objectiveFunction.c

    def delta_large_enough(self, delta):
        return delta > float("inf")

    @property
    def delta_x_hat(self):
        return self.ifs.delta

    @property
    def x_hat(self):
        return self.ifs.x

    def x_bar_hat(self):
        return self.x_bar_hat

    def solve(self):
        # start with centering step (p. 5, section 2.1)

        # Begin Step2
        delta_prev = 0
        # delta_x_hat, touching_constraints = self.constraints.deltaAndTouchingConstraints(self.ifs)
        if not self.delta_large_enough(self.delta_x_hat):
            # 2ii: implementation detail page 6
            d0 = self.ifs.average_direction_of_touching_constraints()
            d00 = d0 - self.objectiveFunction.c * (np.dot(self.objectiveFunction.c, d0) / self.objectiveFunction.norm_sqd)
            #now solve 2-var LP
            a = [[ constraint.norm, -np.dot(constraint.a, d00)] for constraint in self.constraints.constraints]
            b = [np.dot(constraint.a, self.ifs.x) - constraint.b for constraint in self.constraints.constraints]
            c = [1, 0]
            two_var_LP  = TwoVariableLP(a, b, c)
            self.imp_detail_delta, self.imp_detail_alpha = two_var_LP.optimal_solution()
            alpha = self.imp_detail_alpha
            self.setIFS(self.ifs.x + alpha * d00)
        delta = self.ifs.delta
        if delta == 0:
            return 'Finished at step 2iii. '
        # 2. Substep Continued, page 6, our step 2iii
        # Get T(x_hat), x_bar_hat,  x_bar_hat_i for each touch point
        touch_points = self.ifs.touchingPoints
        x_bar_hat = self.ifs.x
        x_bar_hat_i = [ self.objectiveFunction.project_point_to_plane_through_another_point(tp.touch_point, x_bar_hat) for tp in touch_points]
        # Step 2 iv
        print(x_bar_hat_i)


class Constraint:
    # Assume constraint is inequality of form ax >= b
    def __init__(self, a_, b_):
        self.a = np.array(a_)
        self.b = b_
        self.norm_sqd = norm_sqd(self.a)
        self.norm = self.norm_sqd ** 0.5

    def __str__(self):
        return str(self.a) + 'x >= ' + str(self.b)
        
    def directedDistanceFrom(self, x):
        return (np.inner(self.a, x) - self.b) / self.norm
    
    def project(self, x):
        return x - self.a * self.directedDistanceFrom(x) / self.norm

class Constraints:
    def __init__(self, constraints_):
        self.constraints = constraints_[0:]
        # self.A = np.matrix([constraint.a for constraint in constraints_])
        # self.b = np.array([constraint.b for constraint in constraints_])
        
    def __str__(self):
        s = 'constraints:'
        for con in self.constraints:
            s += '\n' + str(con)
        return s
    #
    # def deltaAndTouchingConstraints(self, ifs):
    #     distances = [con.directedDistanceFrom(ifs.x) for con in self.constraints]
    #     delta = min(distances)
    #     distWithConstraint = zip(distances,self.constraints)
    #     return  (delta, [ constraint for dist, constraint in distWithConstraint if dist == delta])
    #
class ObjectiveFunction:
    def __init__(self, c_):
        self.c = np.array(c_)
        self.norm_sqd =  norm_sqd(self.c)
        self.norm =  self.norm_sqd ** 0.5
        self.normed = self.c / self.norm

    def __str__(self):
        return 'minimize ' + str(self.c) + 'x'

    def project_point_to_plane_through_another_point(self, x, plane_point):
        distance_to_point = (np.inner(self.c, x) - np.inner(self.c, plane_point)) / self.norm
        return x - self.c * distance_to_point / self.norm


class TouchPoint:
    def __init__(self, ballCenter, delta, constraint): #, ballBottom, constraints):
        self.ballCenter = ballCenter
        self.delta = delta #yes, this could be computed, but we presumably already have it
        self.constraint = constraint
        self.touch_point = ballCenter - constraint.a * (constraint.a * ballCenter - constraint.b) / constraint.norm ** 2
        # now apply subroutine 1 to get alpha range, which requires a and g as inputs
        # a = constraints.A.dot(self.touchPoint).getA1() - constraints.b
        # g = constraints.A.dot(ballBottom - self.touchPoint).getA1()
        # self.alphaRange = subroutine1(a, g)
        
    def __str__(self):
        return 'touchpoint = ' + str(self.touchPoint) #+', alpha = ' + str(self.alphaRange)


class FeasibleSolution:
    def __init__(self, x_, cons):
        constraints = cons.constraints
        self.x = np.array(x_)
        self.norm =  np.sqrt(np.sum(self.x * self.x))
        distances = [abs(con.directedDistanceFrom(x_)) for con in constraints]
        self.delta = min(distances)
        touching_constraints =  [constraint for dist, constraint in zip(distances, constraints) if dist == self.delta]
        # self.bottomOfBall = self.x -  linearProgram.objectiveFunction.normed * self.delta
        # self.touchingPoints = [TouchPoint(self.x, self.bottomOfBall, self.delta, constraint, linearProgram.constraints) for constraint in self.touchingConstraints]
        self.touchingPoints = [TouchPoint(self.x, self.delta, constraint) for constraint in touching_constraints]

    def __str__(self):
        return 'feasible solution:\n x=' + str(self.x) \
            + '\n delta = ' + str(self.delta) \
            + '\n # touching points = '  \
            + '\n'.join([str(tp) for tp in self.touchingPoints])

    def average_direction_of_touching_constraints(self):
        return sum([touch_point.constraint.a for touch_point in self.touchingPoints]) / float(len(self.touchingPoints))


class TwoVariableLP:
    # solving max cx s/t ax <= b
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def optimal_solution(self):
        m = Model("lp")
        # Create variables
        x = m.addVar(vtype=GRB.CONTINUOUS, name="x")
        y = m.addVar(vtype=GRB.CONTINUOUS, name="y")
        # Set objective
        m.setObjective(x * self.c[0] + y * self.c[1], GRB.MAXIMIZE)

        p = 1
        for a, b in zip(self.a, self.b):
            m.addConstr(x * a[0] + y * a[1] <= b, "c"+str(p))
            p += 1
        m.optimize()
        return [v.x for v in m.getVars()]


