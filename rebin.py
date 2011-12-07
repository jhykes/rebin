"""
Rebin histogram values.

"""

import numpy as np
from numpy.random import uniform

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



def edge_step(x, y, **kwargs):
    """
    Convenience function to plot a histogram with edges and 
    bin values precomputed. The normal matplotlib hist function computes
    the bin values internally.

    Input
    -----
     * x : n+1 array of bin edges.
     * y : n array of histogram values.
    """
    return plt.plot(x, np.hstack([y,y[-1]]), drawstyle='steps-post', **kwargs)


def rebin(x1, y1, x2, interp_kind='piecewise_constant'):
    """
    Rebin histogram values y1 from old bin edges x1 to new edges x2.

    Input
    -----
     * x1 : m+1 array of old bin edges.
     * y1 : m array of old histogram values. This is the total number in 
              each bin.
     * x2 : n+1 array of new bin edges.
     * interp_kind : how is the underlying unknown continuous distribution
                      assumed to look: {'cubic', 'piecewise_constant'}

    Returns
    -------
     * y2 : n array of rebinned histogram values.

    Bins in x2 that are entirely outside the range of x1 are assigned 0.
    """

    if interp_kind == 'piecewise_constant':
        return rebin_piecewise_constant(x1, y1, x2)
    else:
        return rebin_spline(x1, y1, x2, interp_kind=interp_kind)
     

def rebin_spline(x1, y1, x2, interp_kind='piecewise_constant'):
    """
    Rebin histogram values y1 from old bin edges x1 to new edges x2.

    Input
    -----
     * x1 : m+1 array of old bin edges.
     * y1 : m array of old histogram values. This is the total number in 
              each bin.
     * x2 : n+1 array of new bin edges.
     * interp_kind : how is the underlying unknown continuous distribution
                      assumed to look: {'cubic'}

    Returns
    -------
     * y2 : n array of rebinned histogram values.

    The rebinning algorithm assumes that the counts in each old bin are
    uniformly distributed in that bin.

    Bins in x2 that are entirely outside the range of x1 are assigned 0.
    """

    # midpoints of x1
    x1_mid = x1[:-1] + np.ediff1d(x1)

    # constructing data for spline
    xx = np.hstack([x1[0], x1_mid, x1[-1]])
    yy = np.hstack([y1[0], y1, y1[-1]])



def rebin_piecewise_constant(x1, y1, x2):
    """
    Rebin histogram values y1 from old bin edges x1 to new edges x2.

    Input
    -----
     * x1 : m+1 array of old bin edges.
     * y1 : m array of old histogram values. This is the total number in 
              each bin.
     * x2 : n+1 array of new bin edges.

    Returns
    -------
     * y2 : n array of rebinned histogram values.

    The rebinning algorithm assumes that the counts in each old bin are
    uniformly distributed in that bin.

    Bins in x2 that are entirely outside the range of x1 are assigned 0.
    """

    x1 = np.asarray(x1)
    y1 = np.asarray(y1)
    x2 = np.asarray(x2)
    
    # Divide y1 by bin widths.
    #  This converts y-values from bin total to bin average over bin width.
    x1_bin_widths = np.ediff1d(x1)
    y1_ave = y1 / x1_bin_widths
    
    # allocating y2 vector
    n  = x2.size - 1
    y2 = []
    
    # loop over all new bins
    for i in range(n):
        x2_lo, x2_hi = x2[i], x2[i+1]
    
        i_lo, i_hi = np.searchsorted(x1, [x2_lo, x2_hi])
    
        # new bin out of x1 range
        if i_hi == 0 or i_lo == x1.size:
            y2.append( 0. )
            continue
    
        # new bin totally covers x1 range
        elif i_lo == 0 and i_hi == x1.size:
            sub_edges = x1
            sub_dx    = np.ediff1d(sub_edges)
            sub_y_ave = y1_ave
    
        # new bin overlaps lower x1 boundary
        elif i_lo == 0:
            sub_edges = np.hstack( [ x1[i_lo:i_hi], x2_hi ] )
            sub_dx    = np.ediff1d(sub_edges)
            sub_y_ave = y1_ave[i_lo:i_hi]
    
        # new bin overlaps upper x1 boundary
        elif i_hi == x1.size:
            sub_edges = np.hstack( [ x2_lo, x1[i_lo:i_hi] ] )
            sub_dx    = np.ediff1d(sub_edges)
            sub_y_ave = y1_ave[i_lo-1:i_hi]
    
        # new bin is enclosed in x1 range
        else:
            sub_edges = np.hstack( [ x2_lo, x1[i_lo:i_hi], x2_hi ] )
            sub_dx    = np.ediff1d(sub_edges)
            sub_y_ave = y1_ave[i_lo-1:i_hi]
    
        y2.append( (sub_dx * sub_y_ave).sum() )

    return np.array(y2)


if __name__ == '__main__':
    # demo rebin() ---------------------------------------------------

    # old size
    m = 18
    
    # new size
    n = 30
    
    # bin edges 
    x_old = np.linspace(0., 1., m+1)
    x_new = np.linspace(-0.01, 1.02, n+1)
    
    # some arbitrary distribution
    y_old = np.sin(x_old[:-1]*np.pi) / np.ediff1d(x_old)
    
    # rebin
    y_new = rebin(x_old, y_old, x_new)
    
    # plot results ----------------------------------------------------
    import matplotlib.pyplot as plt

    plt.figure()
    edge_step(x_old, y_old, label='old')
    edge_step(x_new, y_new, label='new')
    
    plt.legend()
    plt.title("bins' totals")
    
    plt.figure()
    edge_step(x_old, y_old/np.ediff1d(x_old), label='old')
    edge_step(x_new, y_new/np.ediff1d(x_new), label='new')
    
    plt.legend()
    plt.title("bins' averages")
    
    plt.show()
