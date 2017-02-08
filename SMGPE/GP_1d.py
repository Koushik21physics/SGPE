import numpy as np
from matplotlib.pyplot import *
from roots_test import h_roots_recursive
from roots_test import hermite_on
import odespy as os

n = 20
lamda = 100
T = 100
dt = 0.005
steps = T/0.005
h = 1
space_grid = 256
x_space = np.linspace(-8,8,space_grid)


C_init = np.zeros(n+1)+ 0j
C_init[0] = 1

x1, w1 = h_roots_recursive(2*n+1)

#C = np.zeros((n+1, steps+1))+ 0j
P_nk = np.zeros((n+1,2*n+1))+ 0j

eProots = np.exp(-(x1**2.)/2)
weights = w1 * eProots

for i in xrange(n+1):
    P_nk[i,] = hermite_on(i)(x1) * eProots

psi = np.dot(P_nk.T, C_init)

xi = weights * abs(psi)**2. * psi
effC = np.dot(P_nk, xi)

def f(u,t):
    return -1j* (h*u+ lamda * effC)

solver = os.RK4(f, complex_valued = True)
solver.set_initial_condition(C_init)

time_points = np.linspace(0,T, steps)
u, t = solver.solve(time_points)

def phi(n, x_space):
    phi_n = np.zeros((n+1,space_grid)) + 0j
    for i in xrange(n+1):
        phi_n[i,:] = hermite_on(i)(x_space)*np.exp(-(x_space**2.))
    return phi_n

Phii = phi(n, x_space)

#print np.shape(Phii)
#print np.shape(u)

wave_func = np.dot(u,Phii)
#wave_func = np.sum( u* Phii, axis =1)
density = np.abs(wave_func)**2.


#plot(time_points, density, 'r-')

plot(x_space, density[steps-1,:], 'b-')

#savefig('lamda=1000.png')
show()

#print np.shape(u)
