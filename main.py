import numpy as np
import math as m
import matplotlib.pyplot as plt
import scipy.ndimage.filters as filters
import cython_finite_diff_lap as laplacian

sqrt = math.sqrt
npr = np.random



def sgpe(dt, Nx, Ny, Nz, Nt):
	
	xx = np.linspace(-1,1,Nx)
	yy = np.linspace(-1,1,Ny)
	zz = np.linspace(-1,1,Nz)

	alpha = np.zeros((Nx, Ny, Nz, Nt))+ 0j
	space = np.mgrid(-1:1:Nx*1j,-1:1:Ny*1j,-1:1:Nz*1j)
	space = space.reshape(3,-1).T

	alpha[:,:,:,0] = np.exp(-space**2. /2)
	d2alpha = laplacian.laplacianFD3dcomplex(alpha, xx[1]-xx[0], yy[1]-yy[0], zz[1]-zz[0])

	pot = space**2 /2.0

	
	for i in range(1,Nt):
		d2alpha = laplacian.laplacianFD3dcomplex(alpha[:,:,:,i], xx[1]-xx[0], yy[1]-yy[0], zz[1]-zz[0])
		alpha[:,:,:,i] = (-1j-1/(kb*T))* (-d2alpha[:,:,:,i-1] + pot* alpha[:,:,:,i-1]) *dt + sqrt(dt)* np.random.normal(0,1)
	
	return alpha
