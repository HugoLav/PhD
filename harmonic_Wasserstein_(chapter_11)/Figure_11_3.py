""" 
Code used to produce Figure 11.3 of the manuscript
"""


#********************************************************************************************************
# Package importation 
#********************************************************************************************************

print("Package importation...\n")

# Mathematical functions
import numpy as np
from numpy import linalg as lin
from math import *

# Plotting of the results 
import matplotlib.pyplot as plt

# Program to solve the problem 
import Harmonic_BB_2D

#********************************************************************************************************
# Parameters 
#********************************************************************************************************

# Discretization of the starting space Omega (for the centered grid) 
nO = 13

# Discretization of the target space D
nD = 30

# Number of iterations
Nit = 1000


#********************************************************************************************************
# Building the input for the algorithm  
#********************************************************************************************************


mub0 = np.ones((nO+1,nO,nD,nD)) / (nD*nD)
mub1 = np.ones((nO,nO+1,nD,nD)) / (nD*nD)


# Domain Omega and D --------------------------------------------------

# Domain Omega: centered grid 
xOmegaC = np.linspace(0,1,nO)
yOmegaC = np.linspace(0,1,nO)
xGridOmegaC, yGridOmegaC = np.meshgrid(xOmegaC, yOmegaC)


# Step Omega
DeltaOmega = xOmegaC[1] - xOmegaC[0]

# Domain Omega: staggered grid 
xOmegaS = np.linspace(- DeltaOmega/2,1 + DeltaOmega/2 ,nO+1) 
yOmegaS = np.linspace(- DeltaOmega/2,1 + DeltaOmega/2 ,nO+1)
xGridOmegaS, yGridOmegaS = np.meshgrid(xOmegaS, yOmegaS)


xDC = np.linspace(0,1 - 1/nD,nD)
yDC = np.linspace(0,1 - 1/nD,nD)
xGridDC, yGridDC = np.meshgrid(xDC, yDC, indexing = 'ij')

# Step D
DeltaD = xDC[1] - xDC[0]

# Routines for  Gaussians ------------------------------------------------------------------------

def gaussian( x, covariance) : 
	argument = 0
	for i in range(2) : 
		for j in range(2) : 
			argument += (x[i] - 0.5) * covariance[i,j] * ( x[j] - 0.5) 
	return exp(- argument / 2)

	
def gaussianAngle(x, angle) : 	
	covariance = np.zeros( (2,2) ) 
	covariance[0,0] = 32 * cos(angle)**2 + 8*sin(angle)**2 
	covariance[0,1] =  24 * cos(angle) * sin(angle)
	covariance[1,0] =  24 * cos(angle) * sin(angle)
	covariance[1,1] = 8*cos(angle)**2 + 32*sin(angle)**2
	return gaussian(x, covariance)



# Routines about polar coordinates ---------------------------------
 
def cartesianToPolar(x,y) : 
	x -= 0.5
	y -= 0.5
	if x == 0 : 
		if y > 0 : 
			return pi / 2
		else : 
			return - pi / 2
	elif x < 0 : 
		return np.arctan(y/x) + pi 
	elif x > 0 :
		return np.arctan(y/x) 
			

# Building the boundary conditions ---------------------------------------

for i in range(nO) : 
	for j in range(nD) : 
		for k in range(nD) : 
			
			x = np.zeros( 2 )
			x[0] = xDC[j]
			x[1] = yDC[k]
			
			# For mub0
			
			xO = xOmegaS[0]
			yO = yOmegaC[i]
			angle = cartesianToPolar(yO,xO) 
			mub0[0,i,j,k] = gaussianAngle( x,-angle)
			
			xO = xOmegaS[-1]
			yO = yOmegaC[i]
			angle = cartesianToPolar(yO,xO) 
			mub0[-1,i,j,k] = gaussianAngle( x,-angle)
			
			# For mub1
			
			xO = xOmegaC[i]
			yO = yOmegaS[0]
			angle = cartesianToPolar(yO,xO) 
			mub1[i,0,j,k] = gaussianAngle( x,-angle)
			
			xO = xOmegaC[i]
			yO = yOmegaS[-1]
			angle = cartesianToPolar(yO,xO) 
			mub1[i,-1,j,k] = gaussianAngle( x,-angle)

#********************************************************************************************************
# Calling the main function 
#********************************************************************************************************

mu, E00, E10, E01, E11, objectiveValue, primalResidual, dualResidual = Harmonic_BB_2D.harmonicBB2D(nO, nD, Nit, mub0, mub1,  False )

#********************************************************************************************************
# Plotting the results  
#********************************************************************************************************
	
# First check that convergence indeed happens 
	
plt.plot(objectiveValue[10:])
plt.show()	

plt.semilogy( primalResidual )
plt.semilogy( dualResidual )
plt.show()		


# Then plot the densities --------------------------------------------

numberDisplay = nO
rhoMax = np.max(mu)
rho = rhoMax - mu 

fig, axes = plt.subplots(nrows=numberDisplay, ncols=numberDisplay, figsize = (numberDisplay,numberDisplay) )
	
for i in range(numberDisplay) :
	for j in range(numberDisplay) :
		axes[i,j].imshow( rho[i,j,:,:], cmap=plt.cm.gray, interpolation = 'none', vmin = 0.0, vmax = rhoMax )
		axes[i,j].get_xaxis().set_visible(False)
		axes[i,j].get_yaxis().set_visible(False)

plt.show()	
# plt.savefig("output.png")	

