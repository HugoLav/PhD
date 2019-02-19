"""
We compute the plot on the right in Figure 10.1. We rely on the values of the eigenvalues which are stored in "data_gaussian_explicit.txt", which was computed wih the help of "Figure_10_1_left.py". 

The coordinates of the points where to plot the ellipses are stored in "disk.off". 
"""

# Mathematical functions
import numpy as np
from math import *
from numpy import linalg as lin

# Affichage des résultats 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse

# Import my own methods
from read_off import readOff

"""
Print the result for the gaussians. To put in the manuscript 
"""

#********************************************************************************************************
# Read the file 
#********************************************************************************************************

fileObject = open('data_gaussian_explicit.txt', 'r')

fileObject.readline()

ligne = " "

# Values of the eigenvalues and the radisu
kappa_1 = []
kappa_2 = []
radii = []

ligne = fileObject.readline()

while len(ligne) > 0 : 
	
	decomposition = ligne.split()
	
	radii.append( float(decomposition[0]) )
	kappa_1.append( float(decomposition[1]) )
	kappa_2.append( float(decomposition[2]) )

	ligne = fileObject.readline()
	
fileObject.close()
	
radii = np.array(radii)	
kappa_1 = np.array(kappa_1)
kappa_2 = np.array(kappa_2)

N = len(kappa_1)

#********************************************************************************************************
# Produce the matrix field  
#********************************************************************************************************

# Open the mesh ------------------------------------------------------- 

nameFileMesh = "disk.off"

vertices, triangles, edges = readOff(nameFileMesh)

nVertex = vertices.shape[0]

rMax = 0.
for vertex in range(nVertex) : 
	rMax = max( rMax, lin.norm(vertices[vertex,:] ) )

# Compute the matrix field -----------------------------------------------

def rotation(theta) : 
	return np.matrix([[cos(theta), sin(theta)], [-sin(theta), cos(theta)]])

# Matrix valued field 
matrixField = np.zeros((nVertex,2,2)) 


for vertex in range(nVertex) : 	
	
	x = vertices[vertex,0]
	y = vertices[vertex,1]

	# Compute the radius
	radius = sqrt( x**2 + y**2 ) / rMax
	
	# Find the closest value  			
	indexClosest = ceil((N-1) * radius) 
		
	if x > 0 : 
		angle = atan( y/(x + 10**(-5) ) )
	else : 
		angle = atan( y/(x - 10**(-5) ) ) + pi 
		
	matrixField[vertex,:,:] = np.dot(  rotation(-angle), np.dot( np.diag([kappa_2[indexClosest],kappa_1[indexClosest]]), rotation(angle) ) )	

#********************************************************************************************************
# Plot the results 
#********************************************************************************************************

			
def convertEllips(C, center) : 
	"""
	Plot the ellips whose covariance matrix is C and which is centered at center
	"""
	
	lambda_, v = lin.eig(C)
	
	if v[0,0] > 0 :
		angleEllipse = atan(  v[1,0] / (v[0,0] + 10**(-5) ) )
	else : 
		angleEllipse = atan(  v[1,0] / (v[0,0] - 10**(-5) ) ) + pi
	
	ell = Ellipse(xy=(center[0], center[1]), width=lambda_[0], height=lambda_[1], angle=np.rad2deg(angleEllipse) )
	ell.set_facecolor('0.8')
	ell.set_edgecolor('k')
	
	return ell

# Vizualization 

scaleFactor = 0.03

fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})

for vertex in range(nVertex) : 

	e = convertEllips( scaleFactor* matrixField[vertex,:,:], vertices[vertex, :-1] / rMax  )
	ax.add_artist(e)
	e.set_clip_box(ax.bbox)	
	

ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)

plt.savefig("gaussian_explicit.png")
plt.show()	 

