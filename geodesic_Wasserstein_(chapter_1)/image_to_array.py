"""

Convert an png image into an array
 

"""

# Importation 

import matplotlib.pyplot as plt
import numpy as np

def imageToArray(name, nD) : 
	"""
	name is the name of the png file 
	The image will be a nD*nD array 
	Return a np array 
	"""
	
	input = plt.imread(name)
	inputShape = input.shape

	output = np.zeros((nD,nD))

	StepxD = inputShape[0] // nD 
	StepyD = inputShape[1] // nD 

	for i in range(nD) : 
		for j in range(nD) : 
		
			counterImage = 0 
			toAdd = 0.0
		
			counterI = i*StepxD 
			counterJ = j*StepyD 
		
			while counterI < (i+1) * StepxD : 
				while counterJ < (j+1) * StepyD : 
				
					# Deal with the image 
				
					toAdd += 1.0 - 0.299 * input[counterI, counterJ,0] - 0.587 * input[counterI, counterJ,1] - 0.114 * input[counterI, counterJ,2] 
					counterImage += 1
				
					# Deal with the counters 
				
					counterJ += 1
				
				counterI += 1
			
			output[i,j] = toAdd / counterImage
			
	# Normalization 
	
	output -= np.min(output)
	
	output /= np.sum(output)
	
	return output
			
			
			
			