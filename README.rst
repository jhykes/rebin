This rebin function resamples a 1D or 2D histogram to new bins.

In the 1D case, we have an array ``x1`` of bin edges (``m+1`` entries), and
counts in each one are recorded in array ``y1`` (``m`` entries). Instead of
keeping the data in the ``x1`` bins, we have another set of bins that we want
the data sorted into. This new set of bins is represented by ``x2`` (with
``n+1`` entries).  The rebin function redistributes the counts in ``y1`` into a
new array ``y2`` (``n`` entries). 

To do this rebinning, some assumption about the distribution of the counts
within each channel is necessary. This script offers the choice between a
uniform distribution or a spline fit with specified order.

The function works with array-like objects as determined by Numpy.

Uncertainties in ``y1`` can be propagated through rebin if ``y1`` is a uarray
from the Python uncertainties module.

Knoll[1] describes this in Chapter 18.IV.B titled "Spectrum Alignment."
He calls this process rebinning, relocating, or spectrum alignment.
 
References
----------

 [1] Glenn Knoll, Radiation Detection and Measurement, third edition,
     Wiley, 2000.
