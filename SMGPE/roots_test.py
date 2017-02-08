import numpy
from scipy.misc import factorial
from scipy.special.orthogonal import h_roots, hermite

def hermite_on(n):
	"""Returns orthonormal Hermite polynomial"""
	def func(x):
		return hermite(n)(x) / (numpy.pi ** 0.25) / numpy.sqrt(float(2 ** n * factorial(n)))

	return func

def h_roots_recursive(n):
	"""Returns grid and weights for Gauss-Hermite quadrature"""
	EPS = 1.0e-16
	PIM4 = numpy.pi ** (-0.25) # 0.7511255444649425
	MAXIT = 20 # Maximum iterations.

	x = numpy.empty(n)
	w = numpy.empty(n)
	m = (n + 1) / 2

	z = 0

	for i in xrange(m):
		if i == 0: # Initial guess for the largest root.
			z = numpy.sqrt(float(2 * n + 1)) - 1.85575 * float(2 * n + 1) ** (-0.16667)
		elif i == 1:
			z -= 1.14 * float(n) ** 0.426 / z
		elif i == 2:
			z = 1.86 * z + 0.86 * x[0]
		elif i == 3:
			z = 1.91 * z + 0.91 * x[1]
		else:
			z = 2.0 * z + x[i - 2]

		for its in xrange(MAXIT):
			p1 = PIM4
			p2 = 0.0
			p3 = 0.0
			for j in xrange(n):
				p3 = p2
				p2 = p1
				p1 = z * numpy.sqrt(2.0 / (j + 1)) * p2 - numpy.sqrt(float(j) / (j + 1)) * p3

			pp = numpy.sqrt(float(2 * n)) * p2
			z1 = z
			z = z1 - p1 / pp
			if abs(z - z1) <= EPS:
				break

		if its >= MAXIT:
			raise Exception("Too many iterations")

		x[n - 1 - i] = z
		x[i] = -z
		w[i] = 2.0 / (pp ** 2)
		w[n - 1 - i] = w[i]

	return x, w


if __name__ == '__main__':

	N = 20 # number of points used for integration
	M = 50 # number of mode to extract

	x1, w1 = h_roots(N) # default scipy roots and weights
	x2, w2 = h_roots_recursive(N) # recursive algorithm

	# Eigenfunction of harmonic oscillator (Hermite function)
	def eigenfunction(n):
		def func(x):
			return hermite_on(n)(x) * numpy.exp(-(x ** 2) / 2)
		return func

	# Test function: sum of harmonic oscillator modes (Hermite functions) from 0 to M
	# with coefficients equal to 1.0
	def f(x):
		res = numpy.zeros_like(x)
		for i in xrange(M + 1):
			res += eigenfunction(i)(x)
		return res

	# Extract coefficient for M-th mode from function by integrating
	# C_M = \int f(x) ef(x) dx
	# using Gauss-Hermite quadrature
	# (weights changed because we have exp(-x**2) both in f(x) and ef(x))
	i1 = numpy.sum(f(x1) * w1 * numpy.exp(x1 ** 2) * eigenfunction(M)(x1))
	i2 = numpy.sum(f(x2) * w2 * numpy.exp(x2 ** 2) * eigenfunction(M)(x2))

	# When number of mode to be extracted is high, Scipy roots start to fail
	print "Scipy roots:", x1
	print "Recursive roots:", i2
