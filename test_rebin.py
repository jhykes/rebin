"""
Testing bebin histogram values.

"""

import numpy as np
from numpy.random import uniform


from scipy.optimize import leastsq
from scipy.interpolate import UnivariateSpline

import uncertainties.unumpy as unp

import rebin
from bounded_splines import Bounded_Univariate_Spline


# ---------------------------------------------------------------------------- #
#  Tests for piecewise continuous rebinning
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
def test_x2_same_as_x1():
    """
    x2 same as x1
    """
    # old size
    m = 6
    
    # new size
    n = 6
    
    # bin edges 
    x_old = np.linspace(0., 1., m+1)
    x_new = np.linspace(0., 1., n+1)
    
    # some arbitrary distribution
    y_old = 1. + np.sin(x_old[:-1]*np.pi) / np.ediff1d(x_old)
    
    # rebin
    y_new = rebin.rebin(x_old, y_old, x_new, interp_kind='piecewise_constant')

    assert np.allclose(y_new, y_old)



# ---------------------------------------------------------------------------- #
def test_x2_surround_x1():
    """
    x2 range surrounds x1 range
    """
    # old size
    m = 2
    
    # new size
    n = 3
    
    # bin edges 
    x_old = np.linspace(0., 1., m+1)
    x_new = np.linspace(-0.1, 1.2, n+1)
    
    # some arbitrary distribution
    y_old = 1. + np.sin(x_old[:-1]*np.pi) / np.ediff1d(x_old)
    
    # rebin
    y_new = rebin.rebin(x_old, y_old, x_new, interp_kind='piecewise_constant')

    # compute answer here to check rebin
    y_old_ave  = y_old / np.ediff1d(x_old)
    y_new_here = [y_old_ave[0]*(x_new[1]-0.), 
                  y_old_ave[0]*(x_old[1]-x_new[1]) + y_old_ave[1]*(x_new[2]-x_old[1]),
                  y_old_ave[1]*(x_old[-1]-x_new[-2])]


    assert np.allclose(y_new, y_new_here)
    assert np.allclose(y_new.sum(), y_old.sum())


# ---------------------------------------------------------------------------- #
def test_x2_lower_than_x1():
    """
    x2 range is completely lower than x1 range
    """
    # old size
    m = 2
    
    # new size
    n = 3
    
    # bin edges 
    x_old = np.linspace(0., 1., m+1)
    x_new = np.linspace(-0.2, -0.0, n+1)
    
    # some arbitrary distribution
    y_old = 1. + np.sin(x_old[:-1]*np.pi) / np.ediff1d(x_old)
    
    # rebin
    y_new = rebin.rebin(x_old, y_old, x_new, interp_kind='piecewise_constant')


    assert np.allclose(y_new, [0.,0.,0.])
    assert np.allclose(y_new.sum(), 0.)

# ---------------------------------------------------------------------------- #
def test_x2_above_x1():
    """
    x2 range is completely above x1 range
    """
    # old size
    m = 20
    
    # new size
    n = 30
    
    # bin edges 
    x_old = np.linspace(0., 1., m+1)
    x_new = np.linspace(1.2, 10., n+1)
    
    # some arbitrary distribution
    y_old = 1. + np.sin(x_old[:-1]*np.pi) / np.ediff1d(x_old)
    
    # rebin
    y_new = rebin.rebin(x_old, y_old, x_new, interp_kind='piecewise_constant')


    assert np.allclose(y_new, np.zeros((n,)))
    assert np.allclose(y_new.sum(), 0.)


# ---------------------------------------------------------------------------- #
def test_x2_in_x1():
    """
    x2 only has one bin, and it is surrounded by x1 range
    """
    # old size
    m = 4
    
    # new size
    n = 1
    
    # bin edges 
    x_old = np.linspace(0., 1., m+1)
    x_new = np.linspace(0.3, 0.65, n+1)
    
    # some arbitrary distribution
    y_old = 1. + np.sin(x_old[:-1]*np.pi) / np.ediff1d(x_old)

    
    # rebin
    y_new = rebin.rebin(x_old, y_old, x_new, interp_kind='piecewise_constant')

    # compute answer here to check rebin
    y_old_ave  = y_old / np.ediff1d(x_old)
    y_new_here = (    y_old_ave[1]*(x_old[2]-x_new[0])  
                    + y_old_ave[2]*(x_new[1]-x_old[2]) )

    assert np.allclose(y_new, y_new_here)
    

# ---------------------------------------------------------------------------- #
def test_y1_uncertainties():
    """
    x2 range surrounds x1 range, y1 has uncertainties
    """
    # old size
    m = 2
    
    # new size
    n = 3
    
    # bin edges 
    x_old = np.linspace(0., 1., m+1)
    x_new = np.linspace(-0.1, 1.2, n+1)
    
    # some arbitrary distribution
    y_old = 1. + np.sin(x_old[:-1]*np.pi) / np.ediff1d(x_old)
 
    # with uncertainties
    y_old = unp.uarray( [y_old, 0.1*y_old*uniform((m,))] )
    
    # rebin
    y_new = rebin.rebin(x_old, y_old, x_new, interp_kind='piecewise_constant')

    # compute answer here to check rebin
    y_old_ave  = y_old / np.ediff1d(x_old)
    y_new_here = np.array(
                 [y_old_ave[0]*(x_new[1]-0.), 
                  y_old_ave[0]*(x_old[1]-x_new[1]) + y_old_ave[1]*(x_new[2]-x_old[1]),
                  y_old_ave[1]*(x_old[-1]-x_new[-2])]
                  )


    # mean or nominal value comparison
    assert np.allclose(unp.nominal_values(y_new), 
                       unp.nominal_values(y_new_here))

    # mean or nominal value comparison
    assert np.allclose(unp.std_devs(y_new), 
                       unp.std_devs(y_new_here))
    assert np.allclose(unp.nominal_values(y_new).sum(),
                       unp.nominal_values(y_new_here).sum())


# ---------------------------------------------------------------------------- #
#  Tests for cubic-spline rebinning
# ---------------------------------------------------------------------------- #

def build_spline(x_edges):
    x_mids = x_edges[:-1] + 0.5*np.ediff1d(x_edges)

    xx = np.hstack([x_edges[0], x_mids, x_edges[-1]])

    yy = np.sin(np.pi*xx/xx.max())

    denom = 10
    yy[0] = yy[1]
    #yy[2] = yy[1] / denom
    yy[-2] = yy[-1] 
    #yy[-3] = yy[-1] / denom

    def residual(p):
        yy[0]  = yy[1]  = p[0]
        #yy[2]  = yy[1] / denom
        yy[-1] = yy[-2] = p[1]
        #yy[-3] = yy[-1] / denom 

        spl = Bounded_Univariate_Spline(xx, yy, s=0)

        area0 = spl.integral(xx[0], xx[1])
        area1 = spl.integral(xx[-2], xx[-1])

        return np.array([ p[0] - area0, p[1] - area1])

    p, n_iters = leastsq(residual, [1,1])

    yy[0]  = yy[1]  = p[0]
    yy[-2] = yy[-1] = p[1]

    spl = Bounded_Univariate_Spline(xx, yy, s=0)

    1/0
    return spl

    
    
import matplotlib.pyplot as plt
xf = np.linspace(-0.5, 1.5, 1000)


# ---------------------------------------------------------------------------- #
def test_x2_surrounds_x1():
    """
    x2 range is completely above x1 range
    """
    # old size
    m = 20
    
    # new size
    n = 30
    
    # bin edges 
    x_old = np.linspace(0., 1., m+1)
    x_new = np.linspace(-0.5, 1.5, n+1)
    
    # get spline
    spl = build_spline(x_old)

    y_old = np.array([ spl.integral(x_old[i],x_old[i+1]) for i in range(m) ])

    y_new_ref = np.array(
                 [ spl.integral(x_new[i],x_new[i+1]) for i in range(n) ])

    # rebin
    y_new = rebin.rebin(x_old, y_old, x_new, interp_kind='cubic')


    assert np.allclose(y_new, y_new_ref)
