""" 
Code used to produce Figure 1.1 of the manuscript
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

# Converter of img to array (my own routine) 
import image_to_array

# Program to solve the problem 
import Geodesic_BB_1D 

#********************************************************************************************************
# Parameters 
#********************************************************************************************************

# Discretization of the starting space Omega (for the centered grid) 
nO = 60

# Discretization of the target space D
nD = 100

# Number of iterations
Nit = 10


#********************************************************************************************************
# Building the input for the algorithm  
#********************************************************************************************************


# Congestion constant -------------------------------------------------------------------

cCongestion = 0.0 

# Potential V -----------------------------------------------------------------

potentialV = np.zeros((nD,nD))


# Boundary values ---------------------------------------------------------------  

mub0 = image_to_array.imageToArray("image_one_circle.png", nD)
mub1 = image_to_array.imageToArray("image_two_circles_centered.png", nD)

# Normalization
mub0 /= (np.sum(mub0)) 
mub1 /= (np.sum(mub1)) 


#********************************************************************************************************
# Calling the main function 
#********************************************************************************************************

mu, E0, E1, objectiveValue, primalResidual, dualResidual = Geodesic_BB_1D.geodesicBB1D(nO, nD, Nit, mub0, mub1, cCongestion , potentialV , True )

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


maxMu = np.max(mu)

fig, axes = plt.subplots(nrows=1, ncols=5, figsize = (5,1) )
	
j = 0		
for i in [ 0, (nO//4), (nO//2), (3*nO)//4, nO-1 ] :
	axes[j].imshow( maxMu - mu[i,:,:], cmap=plt.cm.gray, interpolation = 'none', vmin = 0., vmax = maxMu )
	axes[j].get_xaxis().set_visible(False)
	axes[j].get_yaxis().set_visible(False)
	j += 1

plt.show()	
# plt.savefig("output.png")


