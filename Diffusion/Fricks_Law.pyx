import numpy as np
cimport numpy as np
cimport cython
from scipy.optimize import leastsq



DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function


cdef banded_matrix(DTYPE_t a, DTYPE_t b, DTYPE_t c, int n):
    cdef int i
    cdef np.ndarray[DTYPE_t, ndim = 2] matrix = np.zeros((n, n), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim = 1] temp = np.zeros(n, dtype=DTYPE)
    for i in range(n - 1):
        matrix[i][i] = b
        matrix[i][i + 1] = c
        matrix[i + 1][i] = a

    matrix[0][0] = 1
    matrix[0][1] = 0
    matrix[n - 1][n - 2] = 0
    matrix[n - 1][n - 1] = 1

    for i in range(n - 1):
        temp[i] = matrix[i + 1, i] / matrix[i, i]
        matrix[i + 1, :] = matrix[i + 1, :] - matrix[i, :] * temp[i]
    return matrix, temp


cdef np.ndarray[DTYPE_t, ndim = 2] banded_matrix2(DTYPE_t a, DTYPE_t b, DTYPE_t c, int n):
    cdef int i
    cdef np.ndarray[DTYPE_t, ndim = 2] matrix = np.zeros((n, n), dtype=DTYPE)

    for i in range(n - 1):
        matrix[i][i] = b
        matrix[i][i + 1] = c
        matrix[i + 1][i] = a


    matrix[n - 1][n - 1] = b

    return matrix


def ADI(DTYPE_t dx, DTYPE_t dt, int N, DTYPE_t D, FRAP):

    cdef int x_len = FRAP.x2 - FRAP.x1

    cdef np.ndarray[DTYPE_t, ndim=3] u = np.zeros((N, x_len, x_len), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] temp = np.zeros((x_len, x_len), dtype = DTYPE)

    cdef DTYPE_t a = (D*dt)/(dx**2)
    cdef DTYPE_t b = a

    cdef np.ndarray[DTYPE_t, ndim=2] banded_right = banded_matrix2(b, 1 - 2 * b, b, x_len)
    cdef np.ndarray[DTYPE_t, ndim=2] banded_left
    cdef np.ndarray[DTYPE_t, ndim=1] temp_v, vector_b
    banded_left, temp_v = banded_matrix(-a, 1 + 2 * a, -a, x_len)
    u[0] = FRAP.arr[0]


    for n in range(N - 1):
        #first step: along the x coordinate at a fixed coordinate x
        for i in range(x_len-1):
            vector_b = np.dot(banded_right, u[n][i, :])
            vector_b[0] = 0
            vector_b[x_len-1] = 0
            temp[i, :] = x_for_gauss(banded_left, temp_v, vector_b, x_len) #implicit for x

        for j in range(x_len-1):

            vector_b = np.dot(banded_right, temp[:, j])
            vector_b[0] = 0
            vector_b[x_len-1] = 0
            u[n + 1][:, j] = x_for_gauss(banded_left, temp_v, vector_b, x_len)

    return  u


cdef np.ndarray[DTYPE_t, ndim=1] x_for_gauss(np.ndarray[DTYPE_t, ndim=2] matrix, np.ndarray[DTYPE_t, ndim=1] temp, np.ndarray[DTYPE_t, ndim=1] d, int n):
    cdef int i
    cdef np.ndarray[DTYPE_t, ndim=1] x = np.zeros(n, dtype=DTYPE)

    for i in range(n - 1):
        d[i + 1] = d[i + 1] - d[i] * temp[i]

    x[n - 1] = d[n - 1]/matrix[n - 1][n - 1]

    for i in range(n - 1):
        x[n - 2 - i] = (1/matrix[n - 2 - i][n - 2 - i])*(d[n - 2 - i] - matrix[n - 2 - i][n - i - 1]*(x[n - 1 - i]))

    return x


cdef error(DTYPE_t D, DTYPE_t dx, DTYPE_t dt, FRAP):
        return np.ravel(ADI(dx, dt, 5, D, FRAP) - FRAP.arr)


def FIT(DTYPE_t dx, DTYPE_t dt, FRAP, DTYPE_t D_0):
    pfit, pcov, infodict, errmsg, success = leastsq(error, D_0, full_output=1)
    return pfit, pcov


def COASRE(DTYPE_t dx, DTYPE_t dt, int N, np.ndarray[DTYPE_t, ndim=1] D_arr, FRAP):

    cdef np.ndarray[DTYPE_t, ndim=1] Out = np.zeros( N, dtype = DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=3] REAL = FRAP.arr

    cdef int N_ = len(FRAP.t_grid)
    cdef int i

    for i in range(N):
        Out[i] = np.sqrt(np.mean(((ADI(dx, dt, N_, D_arr[i], FRAP) - REAL)**2)))

    print(Out)
    return Out



