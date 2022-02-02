import numpy as np
from scipy.optimize import leastsq


def cymm_gaussian(height, center_x, center_y, width):
    """Returns a gaussian function with the given parameters"""
    width = float(width)

    return lambda x, y: height * np.exp(
        -(((center_x - x) / width) ** 2 + ((center_y - y) / width) ** 2) / 2)


def cymm_moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X * data).sum() / total
    y = (Y * data).sum() / total
    col = data[:, int(y)]
    width_x = np.sqrt(np.abs((np.arange(col.size) - x) ** 2 * col).sum() / col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size) - y) ** 2 * row).sum() / row.sum())
    width = (width_x + width_y)/2
    height = data.max()
    return height, x, y, width



def cymm_fit_gaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = cymm_moments(data)
    errorfunction = lambda p: np.ravel(cymm_gaussian(*p)(*np.indices(data.shape)) -
                                       data)
    pfit, pcov, infodict, errmsg, success = leastsq(errorfunction, params, full_output=1)
    N = data.size  # number of points
    n = 5  # number of parameters
    s_sq = (errorfunction(pfit) ** 2).sum() / (N - n)
    cov = pcov * s_sq
    return pfit, cov, pcov

