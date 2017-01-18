import numpy as np
import math as m
import matplotlib.pyplot as plt
import scipy.ndimage.filters as filters


sqrt = math.sqrt
npr = np.random

def deriv2x(gamma):

	
def delta2x(beta):

	return d2x(beta)+d2y(beta)+d2z(beta)


def sgpe(dt, Nx, Ny, Nz, Nt):

	alpha = np.zeros((Nx, Ny, Nz, Nt))+ 0j
	space = np.mgrid(-1:1:100j,-1:1:100j,-1:1:100j)
	space = space.reshape(3,-1).T

	alpha[:,:,:,0] = np.exp(-space**2. /2)
	
			
