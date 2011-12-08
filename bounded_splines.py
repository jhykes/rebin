#!/usr/bin/env

import numpy as np
from scipy.interpolate import UnivariateSpline

class Bounded_Univariate_Spline(UnivariateSpline):
    """
    Interpolatory spline that returns a constant for x outside the
    specified domain.
    """
    def __init__(self, x, y, fill_value=0.0, **kwargs):
        self.bnds = [x[0], x[-1]]
        self.fill_value = fill_value
        UnivariateSpline.__init__(self, x, y, **kwargs)

    def is_outside_domain(self, x):
        x = np.asarray(x)
        return np.logical_or(x<self.bnds[0], x>self.bnds[1])

    def __call__(self, x):
        outside = self.is_outside_domain(x)

        return np.where(outside, self.fill_value, 
                                 UnivariateSpline.__call__(self, x))
        
    def integral(self, a, b):
        # capturing contributions outside domain of interpolation
        below_dx = np.max([0., self.bnds[0]-a])
        above_dx = np.max([0., b-self.bnds[1]])

        outside_contribution = (below_dx + above_dx) * self.fill_value

        # adjusting interval to interpolatory domain
        a_f = np.max([a, self.bnds[0]])
        b_f = np.min([b, self.bnds[1]])

        if a_f >= b_f:
            return outside_contribution
        else:
            return (outside_contribution +
                      UnivariateSpline.integral(self, a_f, b_f) )
