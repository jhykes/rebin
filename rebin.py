"""
Rebin histogram values.

"""

import numpy as np
from numpy.random import uniform

import matplotlib.pyplot as plt

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





def rebin(x1, y1, x2):
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
    y2 = np.zeros((n,))
    
    # loop over all new bins
    for i in range(n):
        x2_lo, x2_hi = x2[i], x2[i+1]
    
        i_lo, i_hi = np.searchsorted(x1, [x2_lo, x2_hi])
    
        # new bin out of x1 range
        if i_hi == 0 or i_lo == x1.size:
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
    
        y2[i] = np.sum(sub_dx * sub_y_ave)

    return y2


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
