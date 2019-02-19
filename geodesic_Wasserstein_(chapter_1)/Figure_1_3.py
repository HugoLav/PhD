""" 
Code used to produce Figure 1.3 of the manuscript
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
nO = 33

# Discretization of the target space D
nD = 50

# Number of iterations
Nit = 1000


#********************************************************************************************************
# Building the input for the algorithm  
#********************************************************************************************************


# Congestion constant -------------------------------------------------------------------

# (Bottom row)
cCongestion = 5.*10**(-2) 
# (Top and middle row)
cCongestion = 0.

# Potential V -----------------------------------------------------------------


# Domain D (grid is "periodic") : centered grid 
xDC = np.linspace(0,1 - 1/nD,nD)
yDC = np.linspace(0,1 - 1/nD,nD)
xGridDC, yGridDC = np.meshgrid(xDC, yDC, indexing = 'ij')

# Step D
DeltaD = xDC[1] - xDC[0]

potentialV = np.zeros((nD,nD))
# Put no potential to get the top row 

for i in range(nD) : 
	for j in range(nD) : 
		distanceToCenter = lin.norm( np.array([xDC[i], xDC[j]]) - np.array([0.5,0.5] ))
		potentialV[i,j] = 0.5 * exp(  - distanceToCenter * 4 )


# Boundary values ---------------------------------------------------------------  

mub0 = image_to_array.imageToArray("image_one_circle.png", nD)
mub1 = image_to_array.imageToArray("image_star_4.png", nD)

# Normalization
mub0 /= (np.sum(mub0)) 
mub1 /= (np.sum(mub1)) 


#********************************************************************************************************
# Calling the main function 
#********************************************************************************************************

mu, E0, E1, objectiveValue, primalResidual, dualResidual = Geodesic_BB_1D.geodesicBB1D(nO, nD, Nit, mub0, mub1, cCongestion , potentialV , False )

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
for i in range(5) :
	axes[i].imshow( maxMu - mu[8*i,:,:], cmap=plt.cm.gray, interpolation = 'none', vmin = 0., vmax = maxMu )
	axes[j].get_xaxis().set_visible(False)
	axes[j].get_yaxis().set_visible(False)
	j += 1

plt.show()	
# plt.savefig("output.png")


