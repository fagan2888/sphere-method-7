'''
Created on Dec 13, 2016

@author: pbenson
'''
import numpy as np
import sphere as sp
# from root.nested.sphere import Constraint

def davidsTwoVarLP():
    A_input = [ [-1.5, -1],
                 [-1.0, -1],
                 [-0.3, -0.5],
                 [-1, 0],
                 [0, -1],
                 [1, 0],
                 [0, 1],
                 ]
    b_input = [ -27, -21, -9, -15, -16, 0, 0]
    A = sp.Constraints([sp.Constraint(a, b) for a, b in zip(A_input, b_input)])
    c = sp.ObjectiveFunction([-130, -100])
    lp = sp.LinearProgram(A, c)
    lp.setIFS([0.025, 0.025])
    lp.solve()

def davids3DThatFailedInCPlusPlus():
    A_input = [ [1.017909293, 0.045521071, -1.467309732],
        [1.825347223,   0.082472929, 0.794901507],
       [0.625621424, 1.557451598, -1.334935207],
        [-0.509018047, -0.432491033, -0.736388102],
        [-0.079643139, -1.077252937, -0.599373374],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [-1,0,0],
        [0,-1,0],
        [0, 0, -1]
                 ]
    b_input = [-0.797279915,
        -0.316550445,
        -0.87242882,
        -0.149113976,
        -0.994068494,
        -0.821903265,
        -0.125182765,
        -0.763750013,
        -0.49058904,
        -0.663605521,
        -0.125896633]
    A = sp.Constraints([sp.Constraint(a, b) for a, b in zip(A_input, b_input)])
    c_input = [1.675493339,-2.162012239,-1.245917643]
    c = sp.ObjectiveFunction(c_input)
    lp = sp.LinearProgram(A, c)
    lp.setIFS([0.0, 0.0, 0.0])
    lp.solve()
    print("Gurobi solution = " + str(sp.ThreeVariableMinGreaterThanLP(A_input, b_input, c_input).optimal_solution()))
    print("Sphere-7 solution = " + str(lp.x_hat))

def davids_random_LP(num_rows, num_cols):
    A_input = np.random.rand(num_rows + 2 * num_cols, num_cols)
    b_input = -np.random.rand(num_rows + 2 * num_cols)
    # add lower and upper bound of box
    for col in range(num_cols):
        row = [0] * num_cols
        row[col] = 1
        A_input[num_rows + 2 * col] = row
        row[col] = -1
        A_input[num_rows + 2 * col + 1] = row
    A = sp.Constraints([sp.Constraint(a, b) for a, b in zip(A_input, b_input)])
    c_input = np.random.rand(num_cols)
    c = sp.ObjectiveFunction(c_input)
    lp = sp.LinearProgram(A, c)
    lp.setIFS([0] * num_cols)
    sp.logResult(lp)
    answer = lp.solve()
    gurobi_problem = sp.NVariableMinGreaterThanLP(A_input, b_input, c_input)
    # if num_cols == 2:
    #     gurobi_problem = sp.TwoVariableMinGreaterThanLP(A_input, b_input, c_input)
    # elif num_cols == 3:
    #     gurobi_problem = sp.ThreeVariableMinGreaterThanLP(A_input, b_input, c_input)
    # else:
    #     print("Stop right there! davids_random_LP only supports 2-3 variables.")
    #     exit(1)
    sp.logResult("Gurobi solution = " + str(gurobi_problem.optimal_solution()))
    gurValue = gurobi_problem.objective_value()
    sp.logResult("Gurobi obj value = " + str(gurValue))
    sp.logResult("Sphere7 result = " + str(answer))
    sp.logResult("Sphere7 solution = " + str(lp.x_hat))
    sph7Value = np.inner(lp.x_hat, c_input)
    sp.logResult("Sphere7 obj value = " + str(sph7Value))
    diff = sph7Value - gurValue
    sp.logResult("Diff obj value = " + str(diff))
    diff_obj_percent = 100 * diff / abs(gurValue)
    sp.logResult("Diff obj % = " + str(diff_obj_percent))
    return diff_obj_percent


def petesSimple3D():
    A_input = [ [-1.0, 0.0, 0.0],
                 [0.0, -1.0, 0.0],
                 [0.0, 0.0 , -1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                 ]
    b_input = [ -1, -1, -1, 0.0, 0.0, 0.0]
    c_input = [-1, -1, -1]
    print("Gurobi solution = " + str(sp.ThreeVariableMinGreaterThanLP(A_input, b_input, c_input).optimal_solution()))
    A = sp.Constraints([sp.Constraint(a, b) for a, b in zip(A_input, b_input)])
    c = sp.ObjectiveFunction(c_input)
    lp = sp.LinearProgram(A, c)
    lp.setIFS([0.99, 0.99, 0.05])
    lp.solve()


if __name__ == '__main__':
    # davidsTwoVarLP()
    # davids3DThatFailedInCPlusPlus()
    # petesSimple3D()
    # np.random.seed(13)
    # davids_random_LP(5, 3)
    # np.random.seed(53)
    num_problems = 50
    for _ in range(num_problems):
        print(int(davids_random_LP(1, 3) + 0.5) )


