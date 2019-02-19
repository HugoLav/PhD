""" 
Code used to produce Figure 11.4 of the manuscript
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
nD = 17

# Number of iterations
Nit = 1000


#********************************************************************************************************
# Building the input for the algorithm  
#********************************************************************************************************


mub0 = np.zeros((nO+1,nO,nD,nD)) 
mub1 = np.zeros((nO,nO+1,nD,nD)) 

# Non constant delta function (hard coded) ------------------------------------------------
# In fact, each dirac mass is 2x2 pixel. 

for alpha in range(nO) : 
	
	mub0[0,alpha, alpha, alpha] = 1
	mub0[0,alpha, (alpha+1) % nD, alpha % nD] = 1
	mub0[0,alpha, alpha % nD, (alpha+1) % nD] = 1
	mub0[0,alpha, (alpha+1) % nD, (alpha+1) % nD] = 1
	
	mub0[-1,alpha, nO-alpha, nO-alpha] = 1
	mub0[-1,alpha, (nO-alpha+1) % nD, nO-alpha% nD] = 1
	mub0[-1,alpha, nO-alpha % nD, (nO-alpha+1) % nD] = 1
	mub0[-1,alpha, (nO-alpha+1) % nD, (nO-alpha+1)% nD] = 1
	
	mub1[alpha, 0, alpha, alpha] = 1
	mub1[alpha, 0, (alpha+1)% nD, alpha% nD] = 1
	mub1[alpha, 0, alpha, (alpha+1)% nD] = 1
	mub1[alpha, 0, (alpha+1)% nD, (alpha+1)% nD] = 1
	
	mub1[alpha, -1, nO-alpha, nO-alpha] = 1
	mub1[alpha, -1, (nO-alpha+1)% nD, nO-alpha] = 1
	mub1[alpha, -1, nO-alpha, (nO-alpha+1)% nD] = 1
	mub1[alpha, -1, (nO-alpha+1)% nD, (nO-alpha+1)% nD] = 1
		

#********************************************************************************************************
# Calling the main function 
#********************************************************************************************************

mu, E00, E10, E01, E11, objectiveValue, primalResidual, dualResidual = Harmonic_BB_2D.harmonicBB2D(nO, nD, Nit, mub0, mub1,  True )

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

