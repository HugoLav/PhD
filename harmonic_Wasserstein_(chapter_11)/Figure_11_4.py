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
nO = 14

# Discretization of the target space D
nD = 40

# Number of iterations
Nit = 1000


#********************************************************************************************************
# Building the input for the algorithm  
#********************************************************************************************************


mub0 = np.zeros((nO+1,nO,nD,nD)) 
mub1 = np.zeros((nO,nO+1,nD,nD)) 

# # Loading the values of the computed geodesics --------------------------------------------------

# The value on the boundaries were computed with the help of the python code in geodesic_Wasserstein\Geodesic_BB_1D

rho_0 = np.loadtxt('values_rho_0.txt').reshape((nO,nD,nD))
rho_1 = np.loadtxt('values_rho_1.txt').reshape((nO,nD,nD))
rho_2 = np.loadtxt('values_rho_2.txt').reshape((nO,nD,nD))
rho_3 = np.loadtxt('values_rho_3.txt').reshape((nO,nD,nD))

for alpha in range(nO) :
	
	mub0[0,alpha,:,:] = rho_3[alpha,:,:]
	mub1[alpha,-1,:,:] = rho_0[alpha,:,:]
	mub0[-1,alpha,:,:] = rho_1[nO-1-alpha,:,:]
	mub1[alpha,0,:,:] = rho_2[nO-1-alpha,:,:]		

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

