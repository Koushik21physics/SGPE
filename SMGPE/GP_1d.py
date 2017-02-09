import numpy as np
from matplotlib import pyplot as plt
from roots_test import h_roots_recursive, hermite_on
import odespy as os

nbase = 40
lamda = 100
T = 100
dt = 0.005
tsteps = T/0.005
h = 1
Nx = 256
x_low = -8.
x_up = 8.
x_grid = np.linspace(x_low,x_up,Nx)

C_init = np.zeros(nbase+1)+ 0j
C_init[0] = 1

x1, w1 = h_roots_recursive(2*nbase+1)

def effOfC(n, x, w, wave_init):
    P_nk = np.zeros((n+1,2*n+1))+ 0j
    eProots = np.exp(-(x**2.)/2)
    weights = w * eProots

    for i in xrange(n+1):
        P_nk[i,] = hermite_on(i)(x1) * eProots

    psi = np.dot(P_nk.T, wave_init)

    xi = weights * abs(psi)**2. * psi
    effC = np.dot(P_nk, xi)
    return effC

effC = effOfC(nbase, x1, w1, C_init)

def rk4(wave_init, h, lamda, eff_C, Tmax, Nt):
    def f(u,t):
        return -1j* (h*u+ lamda * eff_C)

    solver = os.RK4(f, complex_valued = True)
    solver.set_initial_condition(wave_init)

    time_points = np.linspace(0,Tmax, Nt)
    u, t = solver.solve(time_points)
    return u, t

y, t = rk4(C_init, h, lamda, effC, T, tsteps)

def phi(n, x_space):
    phi_n = np.zeros((n+1, len(x_space))) + 0j
    for i in xrange(n+1):
        phi_n[i,:] = hermite_on(i)(x_space)*np.exp(-(x_space**2.))
    return phi_n

Phii = phi(nbase, x_grid)

wave_func = np.dot(y,Phii)
density = np.abs(wave_func)**2.

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.plot(x_grid, density[tsteps-1,:], 'b-')
ax2 = fig.add_subplot(212)
ax2.plot(t, density[:,Nx/2], 'r-')
plt.show()

#wave_func = np.sum( u* Phii, axis =1)
#plot(time_points, density, 'r-')
#savefig('lamda=1000.png')

