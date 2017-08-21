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
    print(sp.ThreeVariableLP(A_input, b_input, c_input).optimal_solution())

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
    # davids3DThatFailedInCPlusPlus()
    petesSimple3D()

