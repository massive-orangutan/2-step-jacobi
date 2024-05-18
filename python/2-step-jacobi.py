import numpy as np
from numpy import linalg as LA
from math import inf


def twoStepJacobi(A, b, x0, alpha, tol1, tol2):
    if alpha <= 0:
        raise Exception("alpha not greater 0")

    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise Exception("A not a square matrix")

    if LA.det(A) == 0:
        raise Exception("A not regular")

    # make diagonal entries 1
    diag = np.diag(A.diagonal())
    A = np.matmul(LA.inv(diag), A)
    b = np.matmul(LA.inv(diag), b)

    R = np.triu(A, k=1)
    D = np.diag(A.diagonal())
    L = np.tril(A, k=-1)

    r = max(abs(x) for x in LA.eigvals(L + R))


    ktotal = 0

    x_old = x0
    m = 0
    err1 = inf
    while err1 >= tol1:
        bm = b + (alpha + r - 1) * x_old
        
        y_old = x_old
        k = 0
        err2 = inf 
        while err2 >= tol2:
            y_new = 1/(alpha + r) * bm - np.matmul( 1/(alpha + r) * (L + R), y_old)
            err2 = LA.norm(y_new - y_old)

            y_old = y_new
            k += 1

        x_new = y_old
        err1 = LA.norm(x_new - x_old)
        
        x_old = x_new
        m += 1

        ktotal += k


    print(f"inner iterations: {m}")
    print(f"outer iterations: {ktotal}")
    print(f"final x: {x_old}")
    print()




if __name__ == "__main__":
    # A1
    twoStepJacobi(np.array([
            [1,1,0],
            [-2,1,1],
            [0,-2,1]]
        ), 
        np.array([2,0,-1]),
        np.array([0,0,0]), 
        2.2, 10**(-5), 10**(-5)
    )

    # A2
    twoStepJacobi(np.array([
            [1, 5/23, 7/23, 14/23, 16/23], 
            [17/24, 1, 1/24, 1/3, 5/8], 
            [11/25, 18/25, 1, 2/25, 9/25], 
            [10/21, 4/7, 19/21, 1, 1/7], 
            [2/11, 3/11, 13/22, 10/11, 1]]
        ),
        np.array([65/23, 65/24, 13/5, 65/21, 65/22]), 
        np.array([0,0,0,0,0]), 
        1.9, 10**(-5), 10**(-5)
    )

    # A3
    twoStepJacobi(np.array([
            [1,5,4,4,3], 
            [1/4, 1, -1/2, 1, -1/4], 
            [0, 2/3, 1, 2/3, 1/3], 
            [3/10, 8/10, 3/10, 1, 9/10], 
            [1, 4, 1, 2, 1]]
        ), 
        np.array([17, 3/2, 8/3, 33/10, 9]), 
        np.array([0, 0, 0, 0, 0]), 
        1.1, 10**(-5), 10**(-5)
    )

    # A4
    twoStepJacobi(np.array([
            [1,1,1,3],
            [-1,1,-1,1],
            [-1,-1,1,1],
            [1,-1,1,1]]
        ), 
        np.array([6,0,0,2]), 
        np.array([0,0,0,0]), 
        0.38, 10**(-5), 10**(-7)
    )
