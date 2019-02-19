"""
We compute a minimizer of a finite difference discretization of the energy (10.15). This is done with a fixed-step gradient descent algorithm.

We decude the evolution of the eigenvalues as a function of r, which we store in a file "data_gaussian_explicit.txt". The left plot in Figure 10.1 precisely displays these eigenvalues.
"""

#********************************************************************************************************
# Importations
#********************************************************************************************************

# Mathematical functions
import numpy as np
from math import *
import matplotlib.pyplot as plt


#********************************************************************************************************
# Parameters
#********************************************************************************************************

# Boundary conditions 
eigenV0 = 5. 
eigenV1 = 1.

# Number of points
N = 100

# Number of iteration
Niter = 5*10**5

# Time step 
tau = 0.001


#********************************************************************************************************
# Build the relevant objects 
#********************************************************************************************************

# Radius
x = np.linspace(0,1,N)

# Space step 
DeltaX = x[1] - x[0] 
	
# Staggered function for x 
xStaggered = np.zeros(N-1) 
for i in range(N-1) : 
	xStaggered[i] = (i + 0.5) * DeltaX
	# xStaggered[i] = 1.

# The eigenvalues will be stored in a 2*N array, with f[0,:] being kappa_1 and f[1:,] being kappa_2.	
	
# Definition of the energy 
def energy(f) : 
	"""
	Return a finite difference discretization of (10.15)
	"""

	output = 0.
	
	output += 1/( 2* DeltaX ) * np.sum(  np.multiply(  xStaggered,  np.square( f[1:,0] - f[:-1,0]) + np.square( f[1:,1] - f[:-1,1]) )  )
	
	output += DeltaX * np.sum(  np.divide(  np.square(  np.square(f[1:,0]) - np.square(f[1:,1]))  , np.multiply( x[1:], np.square(f[1:,0]) + np.square(f[1:,1]))   )  )   
	
	return output

# Definition of the gradient of the energy 
def gradEnergy(f) : 
	"""
	Return the gradient of the energy defined above
	"""
	
	h = np.divide(  np.square( f[:,0]  ) - np.square(  f[:,1] ), np.square( f[:,0]  ) + np.square(  f[:,1] )   )
	
	grad = np.zeros((N,2))
	
	grad[1:-1,0] -= np.multiply(  xStaggered[1:], f[2:,0] - f[1:-1,0] ) / DeltaX + np.multiply(  xStaggered[:-1], f[:-2,0] - f[1:-1,0] ) / DeltaX
	
	grad[1:-1,1] -= np.multiply(  xStaggered[1:], f[2:,1] - f[1:-1,1] ) / DeltaX + np.multiply(  xStaggered[:-1], f[:-2,1] - f[1:-1,1] ) / DeltaX

	grad[1:-1,0] +=	np.divide(4 * DeltaX * np.multiply( f[1:-1,0], h[1:-1]  ) - 2 * DeltaX * np.multiply( f[1:-1,0], np.square(h[1:-1])  ), x[1:-1]) 
	
	grad[1:-1,1] +=	np.divide(4 * DeltaX * np.multiply( f[1:-1,1], - h[1:-1]  ) - 2 * DeltaX * np.multiply( f[1:-1,1], np.square(h[1:-1])  ), x[1:-1]) 
	
	# On ajoute la composante en 0. On suppose que f[0,0] = f[0,1]
	grad[0,0] = xStaggered[0] * (  2 * f[0,0] - f[1,0] - f[1,1]  ) / DeltaX
	
	return grad
	
#********************************************************************************************************
# Gradient descent 
#********************************************************************************************************

# Initialization
f = np.zeros((N,2))

f[:,0] = np.linspace( 2, eigenV0, N)
f[:,1] = np.linspace( 2, eigenV1, N)


# To store the evolution of the energy along the gradient descent iterations. 
energyArray = np.zeros(Niter)
energyArray[0] = energy(f)

# Loop 

for i in range(Niter) : 
	# compute the gradient 
	gradF = gradEnergy( f )
	
	f[1:-1,:] -= tau * gradF[1:-1,:]
	f[0,0] -= tau * gradF[0,0]
	f[0,1] = f[0,0]
	
	energyArray[i] = energy(f)
	
#********************************************************************************************************
# Plot and store the results  
#********************************************************************************************************
	
# Evolution of the energy 
plt.plot( range(Niter), energyArray, '-' )
plt.show()
	
# Eigenvalues as a function of the radius 	
plt.plot(  x, f[:,0], '-', color = 'r'  )
plt.plot(  x, f[:,1], '-', color = 'b'  )
plt.plot(  x, (f[:,0] + f[:,1])/2, '-', color = 'g'  )
plt.show()

# Store the result in a file suited for pgfplots (tikz extension)

fileObject = open('data_gaussian_explicit.txt', 'w') 

fileObject.write("rayon kappa_1 kappa_2 Moyenne\n")

for i in range(N) : 
	fileObject.write( str(x[i]) + " " + str( f[i,1] ) + " " + str( f[i,0] ) + " " + str( (f[i,1] + f[i,0]) /2  ) + "\n" )

fileObject.close()	





