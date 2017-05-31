'''
Created on Dec 13, 2016

@author: pbenson
'''
import numpy as np
import sphere as sp
# from root.nested.sphere import Constraint

if __name__ == '__main__':

# construct an LP, and solve it
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
    c = sp.ObjectiveFunction([-600, -100])
    lp = sp.LinearProgram(A, c)
    print(lp)
    print('\n')
    lp.setIFS([1, 3])
    print(lp.ifs)
    print('\n')
    lp.solve()
#     