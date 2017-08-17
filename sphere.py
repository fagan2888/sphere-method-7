'''
Created on Dec 13, 2016

@author: pbenson
'''

import numpy as np
from gurobipy import Model, GRB

DEBUG = True

def norm_sqd(x):
    return np.sum(x * x)

def subroutine1(a, g):
    # return a 2-tuple that represents range of values satisfying all a-g inequalities
    v1 = float("-inf")
    v1_ratios = [ -float(a_t) / g_t for a_t, g_t in zip(a,g) if g_t > 0]
    if v1_ratios:
        v1 = max(v1_ratios)
    v2 = float("inf")
    v2_ratios = [ (-float(a_t) / g_t) for a_t, g_t in zip(a,g) if g_t < 0]
    if v2_ratios:
        v2 = min(v2_ratios)
    if v1 > v2:
        # return "No solution found in subroutine1"
        raise ValueError
    return (v1, v2)

def bottom_of_ball(ifs, objective_function):
    return ifs.x - (ifs.delta / objective_function.norm) * objective_function.c

def average_vector(vectors):
   return sum(vectors) / float(len(vectors))

def logResult(result):
    if DEBUG:
        print('DEBUG: ' + str(result))

class LinearProgram:
    def __init__(self, constraints, objectiveFunction):
        self.constraints = constraints
        self.objectiveFunction = objectiveFunction
        # self._improvement_threshold_step_2_v =

    def __str__(self):
        return 'Linear Program:\n' + str(self.constraints) + '\n' + str(self.objectiveFunction)
        
    def setIFS(self, x):
        self.ifs = FeasibleSolution(x, self.constraints, self.objectiveFunction)
        self.x_hat_bar = bottom_of_ball(self.ifs, self.objectiveFunction)

    def delta_large_enough(self, delta):
 # TODO
 #  delta > float("inf")
        return True

    @property
    def delta_x_hat(self):
        return self.ifs.delta

    @property
    def x_hat(self):
        return self.ifs.x

    def x_hat_bar(self):
        return self.x_hat_bar

    def epsilon_step_2_v(self):
        return 0.1

    @property
    def max_centering_steps(self):
        return 1

    @property
    def improvement_threshold_step_2_v(self):
        # TODO
        return 0.05 * abs(np.inner(self.objectiveFunction.c, self.x_hat_bar))

    def solve(self):
        # start with centering step (p. 5, section 2.1)
        centering_count = 0
        while True:
            logResult('Centering step # ' + str(centering_count) + '...')
            logResult('IFS = ' + str(self.ifs))

            # Begin Step2 (Centering Step)
            delta_prev = 0
            if not self.delta_large_enough(self.delta_x_hat):
                # TODO Not executing yet, because delta is always large enough
                # 2ii: implementation detail page 6
                logResult("delta_x_hat = " + str(self.delta_x_hat) +", executing implementation detail")
                d0 = self.ifs.average_direction_of_touching_constraints
                d00 = d0 - self.objectiveFunction.c * (np.dot(self.objectiveFunction.c, d0) / self.objectiveFunction.norm_sqd)
                #now solve 2-var LP
                a = [[ constraint.norm, -np.dot(constraint.a, d00)] for constraint in self.constraints.constraints]
                b = [np.dot(constraint.a, self.ifs.x) - constraint.b for constraint in self.constraints.constraints]
                c = [1, 0]
                two_var_LP  = TwoVariableLP(a, b, c)
                self.imp_detail_delta, self.imp_detail_alpha = two_var_LP.optimal_solution()
                logResult('In Centering, delta, alpha = ' + str(self.ifs.x) )
                alpha = self.imp_detail_alpha
                self.setIFS(self.ifs.x + alpha * d00)
            logResult('x_hat_bar = ' + str(self.x_hat_bar))
            delta = self.ifs.delta
            if delta == 0:
                return 'Finished at step 2iii. '
            # 2. Substep Continued, page 6, our step 2iii
            # Get T(x_hat), x_hat_bar,  x_hat_bar_i for each touch point
            touch_points = self.ifs.touchingPoints
            self.x_hat_bar_i = [ self.objectiveFunction.project_point_to_plane_through_another_point(tp.touch_point, self.x_hat_bar) for tp in self.ifs.touchingPoints]

            # Step 2 iv, iterate over touch points, using Subroutine 1 to find all ranges of alphas
            alpha_i2s = []
            for tp in self.ifs.touchingPoints:
                a = []
                for constraint in self.constraints.constraints:
                    x = np.inner(constraint.a, tp.touch_point) - constraint.b
                    a.append(x)
                g = [np.inner(constraint.a, self.x_hat_bar - tp.touch_point) for constraint in self.constraints.constraints]
                alpha_range = subroutine1(a, g)
                if alpha_range[1] == float("inf"):
                    return 'solution diverges to -infinity'
                alpha_i2s.append(alpha_range[1])
            objective_at_x_hat_bar = np.inner(self.x_hat_bar, self.objectiveFunction.c)

            # step 2v
            best_objective_change = None
            for tp, alpha_i2 in zip(self.ifs.touchingPoints, alpha_i2s):
                x_hat_i2 = alpha_i2 * self.x_hat_bar + (1 - alpha_i2) * tp.touch_point
                logResult('x_hat_i2 = ' + str(x_hat_i2))
                objective_change = np.inner(x_hat_i2, self.objectiveFunction.c) - objective_at_x_hat_bar
                if not best_objective_change or objective_change < best_objective_change:
                    best_objective_change = objective_change
                    x_hat_r2 = x_hat_i2
                    x_hat_r = tp.touch_point

            logResult("x_hat_r2 = " + str(x_hat_r2))
            logResult("x_hat_r = " + str(x_hat_r))
            self.x_tilde = self.x_hat_bar + (1 - self.epsilon_step_2_v()) * (x_hat_r2 - self.x_hat_bar)
            logResult("x_tilde = " + str( self.x_tilde))
            logResult("improvement = " + str(-best_objective_change))
            logResult("improvement_threshold = " + str(self.improvement_threshold_step_2_v))
            if -best_objective_change > self.improvement_threshold_step_2_v:
                self.setIFS(self.x_tilde)
            else:

                # Step 2 vi
                if True:
                    previous_center = None
                    previous_opt_delta = float("-inf")
                    centering_count += 1
                    x_tilde_ifs = FeasibleSolution(self.x_tilde, self.constraints, self.objectiveFunction)
                    y_tilde = bottom_of_ball(x_tilde_ifs, self.objectiveFunction)
                    logResult("y_tilde = " + str(y_tilde))
                    y_1 = self.objectiveFunction.project_point_to_plane_through_another_point(x_hat_r, y_tilde)
                    logResult("y_1 = " + str(y_1))
                    y_2 = self.objectiveFunction.project_point_to_plane_through_another_point(x_hat_r2, y_tilde)
                    logResult("y_2 = " + str(y_2))
                    # use subroutine 1 to find feasible range of alphas connecting y_1 and y_2
                    dy = y_2 - y_1
                    a = [np.inner(con.a, y_1) - con.b for con in self.constraints.constraints]
                    g = [np.inner(con.a, dy)  for con in self.constraints.constraints]
                    # TODO alpha1 and alpha2 are not being USED!
                    alpha1, alpha2 = subroutine1(a, g)
                    # # top of page 10!
                    # create 2-var LP, with variables delta and alpha (in that order)
                    b = a
                    # now, redefine a to use it in the LP (presumably for readability!)
                    a = [ [con.norm , - np.inner(con.a, dy)] for con in self.constraints.constraints ]
                    # # add non-negativity constraint
                    c = [1, 0]

                    twoVarLP = TwoVariableLP(a, b, c)
                    opt_delta, opt_alpha = twoVarLP.optimal_solution()
                    if opt_alpha == float("inf"):
                        return "unbounded"
                    x_c_of_alpha = opt_alpha * y_2 + (1 - opt_alpha) * y_1
                    self.setIFS(x_c_of_alpha)
                    logResult("x_c_of_alpha = " + str(x_c_of_alpha))
                    logResult("delta = " + str(self.delta_x_hat))

                    # TODO: per page 10, last para of Approach 2, could continue
                    # if objective improving at "good rate"
                    if opt_delta > previous_opt_delta and centering_count < self.max_centering_steps:
                        previous_opt_delta = opt_delta
                    else:
                        break
        print('Final centering_count = ' + str(centering_count) )

        # Step 2.2 in paper, bottom of page 10. Step 3 in whiteboard algorithm.
        x_bar = x_c_of_alpha
        # if boundary point, we are optimal (done)
        if self.ifs.delta == 0:
            return "done"

        # Descent steps, page 11, (a).
        if True:
            logResult("Beginning Descent, starting from objective value = " + str(self.objectiveFunction.x(self.ifs.x)))
            gamma_set = []

            # Descend in direction of objective function
            gamma_set.append(self.general_descent(- self.objectiveFunction.c, 'obj function'))

            # Descend in average of collection of orthogonal projections of obj function onto each touch point
            c_i_s = []
            for tp in self.ifs.touchingPoints:
                constraint = tp.constraint
                c_i = self.objectiveFunction.c - constraint.a * ((np.inner(constraint.a, self.objectiveFunction.c)/constraint.norm_sqd))
                c_i_s.append(c_i)

            gamma_set.append(self.general_descent(- average_vector(c_i_s), 'avg projected obj function'))

            # Descend in average of touching point normal vectors (* -1 if vector opposes obj function)
            tp_normals_signed = []
            for tp in self.ifs.touchingPoints:
                constraint = tp.constraint
                if np.inner(constraint.a, self.objectiveFunction.c) > 0:
                    tp_normals_signed.append(constraint.a)
                else:
                    tp_normals_signed.append(-constraint.a)

            gamma_set.append(self.general_descent(-average_vector(tp_normals_signed), 'avg touch point'))

            # direction of current center - previous center
            if previous_center:
                gamma_set.append(self.general_descent(x_bar - previous_center, 'center - prev center'))
            else:
                previous_center = x_bar

            # Descent steps D5.1, page 12, (b). Steepest descent parallel to constraint from near touching point.
            if True:
                epsilonD5dot1 = 0.001
                for tp, c_i in zip(self.ifs.touchingPoints, c_i_s):
                    near_touching_point = (1 - epsilonD5dot1) * tp.touch_point + epsilonD5dot1 * self.x_hat
                    gamma_set.append(self.general_descent_from_point(near_touching_point, -c_i,
                                                                     'Steepest descent parallel to constraint from near touching point'))
                    logResult("D5.1 touch_point constraint = " + str(tp.constraint))

            # Descent steps D5.7, page 12, (b). Find bottom of 2D slice.
            if True:
                x_r_2 = None
                x_r_2 = None
                max_touch_count = 0
                max_separation = 0
                for tp in self.ifs.touchingPoints:
                    a = []
                    g = []
                    dx = self.x_hat_bar - tp.projection_to_objective_plane
                    for constraint in self.constraints.constraints:
                        a.append(np.inner(constraint.a, tp.projection_to_objective_plane) - constraint.b)
                        g.append(np.inner(constraint.a, dx))
                    alpha_i1, alpha_i2 = subroutine1(a, g)

                    if alpha_i1 == float("-inf") or alpha_i2 == float("inf"):
                        # Case 1, p 12...skip it for now TODO
                        print("trouble...not implemented")
                    else:
                        # Case 2
                        x_i_1 = alpha_i1 * self.x_hat_bar + (1 - alpha_i1) * tp.projection_to_objective_plane
                        x_i_2 = alpha_i2 * self.x_hat_bar + (1 - alpha_i2) * tp.projection_to_objective_plane
                        # Find set of constraints I2 that are in lower half? TODO
                        # Count if these x_i touch some constraints
                        touched_count = 0
                        tolerance = 0.05
                        for constraint in self.constraints.constraints:
                            if constraint.satisfies(x_i_1, tolerance) or constraint.satisfies(x_i_2, tolerance):
                                touched_count += 1

                        if touched_count >= max_touch_count:
                            if touched_count > max_touch_count:
                                separation = np.linalg.norm(x_i_1 - x_i_2)
                                if separation > max_separation:
                                    x_r_1 = x_i_1
                                    x_r_2 = x_i_2

                if x_r_1 is None:
                    return "Cannot find any endpoints in Case 2"
                # Solve 2-var LP
                x_r_2_minus_x_r_1 = x_r_2 - x_r_1
                # initialize with non-negativity on lambda
                a = [[0, -1]]
                b = [0]
                c = - self.objectiveFunction.c
                for constraint in self.constraints.constraints:
                    a.append([-np.inner(constraint.a, x_r_2_minus_x_r_1),
                              -np.inner(constraint.a, -self.objectiveFunction.c)])
                    b.append(np.inner(constraint.a, x_r_1) - constraint.b)
                page14LP = TwoVariableLP(a, b, c)
                alpha, lamda = page14LP.optimal_solution()
                descend_to = x_r_1 + alpha * x_r_2_minus_x_r_1 - lamda * self.objectiveFunction.c
                direction = descend_to - self.x_hat
                gamma2 = 1
                obj_value = np.inner(self.objectiveFunction.c, descend_to)
                gamma_set.append((obj_value, gamma2, direction , 'Descent to best bottom of slice.'))

            # now find best direction, adjust with epsilon_for_gamma
            best = min(gamma_set)
            logResult("best uses " + best[3])
            epsilon_for_gamma = 0.001
            new_gamma =  best[1] * (1 - epsilon_for_gamma)
            direction = best[2]
            self.setIFS( self.x_hat + new_gamma * direction )
            near_best = (np.inner(self.objectiveFunction.c, self.x_hat + new_gamma * direction), new_gamma, direction, best[3])


        logResult("ifs: " + str(self.ifs.x))
        print('Solving complete')


    def general_descent_from_point(self, point, direction, description):
        gamma2 = float("inf")
        # how far can we go in the given direction?
        for constraint in self.constraints.constraints:
            a_dot_d = np.inner(constraint.a, direction)
            if a_dot_d < 0:
                gamma2 = min (gamma2, (constraint.b - np.inner(constraint.a, point )) / a_dot_d)
        if gamma2 == float("inf"):
            return "Objective function unbounded below"
        return (np.inner(self.objectiveFunction.c, point + gamma2 * direction), gamma2, direction, description)


    def general_descent(self, direction, description):
        return self.general_descent_from_point(self.x_hat, direction, description)




class Constraint:
    # Assume constraint is inequality of form ax >= b
    def __init__(self, a_, b_):
        self.a = np.array(a_)
        self.b = b_
        self.norm_sqd = norm_sqd(self.a)
        self.norm = self.norm_sqd ** 0.5

    def __repr__(self):
        return str(self.a) + 'x >= ' + str(self.b)
        
    def directedDistanceFrom(self, x):
        return (np.inner(self.a, x) - self.b) / self.norm
    
    def project(self, x):
        return x - self.a * self.directedDistanceFrom(x) / self.norm

    def satisfies(self, x, tolerance):
        return (self.directedDistanceFrom(x) / self.norm) < tolerance

class Constraints:
    def __init__(self, constraints_):
        self.constraints = constraints_[0:]
        # self.A = np.matrix([constraint.a for constraint in constraints_])
        # self.b = np.array([constraint.b for constraint in constraints_])
        
    def __repr__(self):
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

    def __repr__(self):
        return 'minimize ' + str(self.c) + 'x'

    def x(self, x):
        return np.inner(self.c, x)

    def project_point_to_plane_through_another_point(self, x, plane_point):
        distance_to_point = (np.inner(self.c, x) - np.inner(self.c, plane_point)) / self.norm
        return x - self.c * distance_to_point / self.norm

class TouchPoint:
    def __init__(self, ballCenter, delta, constraint, objective_func): #, ballBottom, constraints):
        self.ballCenter = ballCenter
        self.delta = delta #yes, this could be computed, but we presumably already have it
        self.constraint = constraint
        constraint_scalar = (np.inner(constraint.a, ballCenter) - constraint.b) / constraint.norm_sqd
        self.touch_point = ballCenter - constraint.a * constraint_scalar
        self.bottom = ballCenter - (self.delta / objective_func.norm) * objective_func.c
        self.projection_to_objective_plane = self.touch_point - np.multiply(np.transpose(objective_func.c), np.inner(objective_func.c, self.touch_point - self.bottom)) / objective_func.norm_sqd
        # now apply subroutine 1 to get alpha range, which requires a and g as inputs
        # a = constraints.A.dot(self.touchPoint).getA1() - constraints.b
        # g = constraints.A.dot(ballBottom - self.touchPoint).getA1()
        # self.alphaRange = subroutine1(a, g)

    def __repr__(self):
        return 'touchpoint = ' + str(self.touch_point) #+', alpha = ' + str(self.alphaRange)

class FeasibleSolution:
    def __init__(self, x_, cons, objective_function):
        constraints = cons.constraints
        self.x = np.array(x_)
        self.norm =  np.sqrt(np.sum(self.x * self.x))
        distances = [abs(con.directedDistanceFrom(x_)) for con in constraints]
        self.delta = min(distances)
        self.touching_constraints =  [constraint for dist, constraint in zip(distances, constraints) if dist / self.delta < 1.000001]
        self.touchingPoints = [TouchPoint(self.x, self.delta, constraint, objective_function) for constraint in self.touching_constraints]
        self.average_direction = average_vector([tp.constraint.a for tp in self.touchingPoints])
    def __repr__(self):
        return 'feasible solution:\n x=' + str(self.x) \
            + '\n delta = ' + str(self.delta) \
            + '\n touching points: '  \
            + '\n'.join([str(tp) for tp in self.touchingPoints])\
               + '\n touching constraints: ' \
               + '\n'.join([str(tc) for tc in self.touching_constraints])

    @property
    def average_direction_of_touching_constraints(self):
        return self.average_direction

class TwoVariableLP:
    # solving max cx s/t ax <= b
    # a is a list of 2-element lists of floats (the rows)
    # b, c is a list of floats
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def optimal_solution(self):
        m = Model("lp")
        m.setParam('OutputFlag', 0)
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
        vars = m.getVars()
        if vars:
            return [v.x for v in vars]
        else:
            raise ValueError


