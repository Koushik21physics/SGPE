import numpy as np
from matplotlib import pyplot as plt
from roots_test import h_roots_recursive, hermite_on
import odespy as os

nbase = 86
lamda = 100
T = 500
dt = 0.0015
tsteps = T/dt
h = np.arange(nbase+1)+ 1/2.
Nx = 100
x_low = -3.
x_up = 3.

x_grid = np.linspace(x_low, x_up, Nx)

C_init = np.zeros(nbase+1)+ 0j
C_init[0] = 1+0j

x1, w1 = h_roots_recursive(2*nbase+1)

def effOfC(n, x, w, wave_init):
    Pnk = np.zeros((n+1,2*n+1))+ 0j
    eProots = np.exp(-(x**2.)/4)
    wts = w * eProots

    for i in xrange(n+1):
        Pnk[i,] = hermite_on(i)(x/np.sqrt(2)) * eProots

    #psi = np.dot(Pnk.T, wave_init)

    #xi = wts * abs(psi)**2. * psi
    #effC = np.dot(Pnk, xi)
    return Pnk, wts 
    #return effC


P_nk, weights = effOfC(nbase, x1,w1, C_init)
#effC = effOfC(nbase, x1, w1, C_init)



def rk4(wave_init, h, lamda, Pnk, wts, Tmax, Nt):
    def f(u,t):
        psi = np.dot(Pnk.T, u)
        xi = wts * abs(psi)**2. * psi
        effC = np.dot(Pnk, xi)
        return -1j* (h*u+ lamda * effC)

    solver = os.RK4(f, complex_valued = True)
    solver.set_initial_condition(wave_init)

    time_points = np.linspace(0,Tmax, Nt)
    u, t = solver.solve(time_points)
    return u, t

y, t = rk4(C_init, h, lamda, P_nk, weights, T, tsteps)

def phi(n, x_space):
    phi_n = np.zeros((n+1, len(x_space))) + 0j
    for i in xrange(n+1):
        phi_n[i,:] = hermite_on(i)(x_space)*np.exp(-(2* x_space**2.))
    return phi_n

Phii = phi(nbase, x_grid)

wave_func = np.dot(y,Phii)
density = (np.abs(wave_func))**2.

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.plot(x_grid, density[tsteps-1,:], 'b-')
ax2 = fig.add_subplot(212)
ax2.plot(t, density[:,Nx/2-1], 'r-')
plt.show()

#wave_func = np.sum( u* Phii, axis =1)
#plot(time_points, density, 'r-')
#savefig('lamda=1000.png')

