#
#  Helper functions for IAML 2021/22 S2 cw2
#   by Hiroshi Shimodaira (h.shimodaira@ed.ac.uk)
#
import os
import gzip
import numpy as np
import scipy
from scipy.io import loadmat
import pandas as pd

def print_versions():
    import platform
    import scipy
    import numpy
    import sklearn
    import matplotlib
    import seaborn
    import pandas

    tabs = [ ['Python', platform.python_version(), '3.9.2'],
             ['Scipy', scipy.__version__, '1.7.0'],
             ['Numpy', numpy.__version__, '1.21.1'],
             ['Sklearn', sklearn.__version__, '0.24.2'],
             ['Matplotlib', matplotlib.__version__, '3.4.2'],
            ]
    for p in tabs:
        print('%s\t%s ' % (p[0], p[1]), end='')
        if p[1] == p[2]:
            print(': Ok')
        else:
            print('<=> %s' % p[2])


#
# Credit: https://stackoverflow.com/questions/44549369/kullback-leibler-divergence-from-gaussian-pm-pv-to-gaussian-qm-qv
#
#  KL-divergence from Q(m1,S1) to P(m0,S0), i.e. KLD(P||Q)
#
def kl_mvn(m0, S0, m1, S1):
    """
    Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
    Also computes KL divergence from a single Gaussian pm,pv to a set
    of Gaussians qm,qv.
    

    From wikipedia
    KL( (m0, S0) || (m1, S1))
         = .5 * ( tr(S1^{-1} S0) + log |S1|/|S0| + 
                  (m1 - m0)^T S1^{-1} (m1 - m0) - N )
    """
    # store inv diag covariance of S1 and diff between means
    N = m0.shape[0]
    iS1 = np.linalg.inv(S1)
    diff = m1 - m0

    # kl is made of three terms
    tr_term   = np.trace(iS1 @ S0)
    det_term  = np.log(np.linalg.det(S1)/np.linalg.det(S0)) #np.sum(np.log(S1)) - np.sum(np.log(S0))
    quad_term = diff.T @ np.linalg.inv(S1) @ diff #np.sum( (diff*diff) * iS1, axis=1)
    #print(tr_term,det_term,quad_term)
    return .5 * (tr_term + det_term + quad_term - N)


