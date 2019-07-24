
import scipy.spatial.kdtree
import numpy
from scipy import version
from numpy import arcsin
import numpy as np
import dask.array as da

scipy_version = ('.'.join(version.version.split('.')[0:2])).split('.')[0:2]

def match_lists(ra1, dec1, ra2, dec2, dist, numNei=1):
    """crossmatches the list of objects (ra1,dec1) with
    another list of objects (ra2,dec2) with the matching radius "dist"
    The routines searches for up to numNei closest neighbors
    the routine returns the distance to the neighbor and the list
    of indices of the neighbor. Everything is in degrees.
    if no match is found the distance is NaN.
    Example:
    > dist, ind = match_lists(ra1,dec1,ra2,dec2, 1./3600)
    > goodmatch_ind = numpy.isfinite(dist)
    > plot(ra1[goodmatch_ind],ra2[ind[goodmatch_ind]])
    Another example:
    > print match_lists( [1,1], [2,3], [1.1,1.2,6,], [2.1,2.2,10], 0.3,numNei=2)
        (array([[ 0.1413761 ,  0.28274768],
                [        inf,         inf]]),
         array([[0, 1],
                [3, 3]]))
    """
    cosd = lambda x: da.cos(x * np.pi/180)
    sind = lambda x: da.sin(x * np.pi/180)
    mindist = 2 * sind(dist/2.)
    getxyz = lambda r, d: [cosd(r)*cosd(d), sind(r)*cosd(d), sind(d)]
    xyz1 = numpy.array(getxyz(ra1, dec1))
    xyz2 = numpy.array(getxyz(ra2, dec2))

    if (int(scipy_version[0])==0) and (int(scipy_version[1])<8):
    # If old scipy version is detected then we use KDTree instead of 
    # cKDTtree because there is a bug in the cKDTree
    # http://projects.scipy.org/scipy/ticket/1178
        tree2 = scipy.spatial.KDTree(xyz2.T)
    else:
        tree2 = scipy.spatial.cKDTree(xyz2.T)
    del xyz2
    ret = tree2.query(xyz1.T, numNei, 0, 2, mindist)
    del xyz1
    dist, ind = ret
    finite = numpy.isfinite(dist)
    dist[finite] = (2*arcsin(dist[finite]/2)) * 180 / np.pi

    return dist, ind
