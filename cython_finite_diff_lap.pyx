from _future_ import divsion
cimport numpy as np
cimport cython
import numpy as np

#3D laplacian of a complex function
@cython.boundscheck(False) # turn of bounds-checking for entire function
def laplacianFD3dcomplex(np.ndarray[double complex, ndim=3] f, double complex dx, double complex dy, double complex dz):
    cdef unsigned int i, j, k, ni, nj, nk
    cdef double complex ifactor, jfactor, kfactor, ijkfactor
    ni = f.shape[0]
    nj = f.shape[1]
    nk = f.shape[2]
    cdef np.ndarray[double complex, ndim=3] lapf = np.zeros((ni,nj,nk)) +0.0J

    ifactor = 1/dx**2
    jfactor = 1/dy**2
    kfactor = 1/dz**2
    ijkfactor = 2.0*(ifactor + jfactor + kfactor)

    for i in xrange(1,ni-1):
        for j in xrange(1, nj-1):
            for k in xrange(1, nk-1):
                lapf[i, j, k] = (f[i, j, k-1] + f[i, j, k+1])*kfactor + (f[i, j-1, k] + f[i, j+1, k])*jfactor + (f[i-1, j, k] + f[i+1, j, k])*ifactor - f[i,j,k]*ijkfactor
    return lapf
