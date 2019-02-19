""" 
Solution of the geodesic problem via a Benamou Brenier method

Starting domain : segment [0,1] 
Target domain D : Torus of 2 dimension

It is a simple an adaptation of the code for the harmonic problem. We have included the possibility to put a potential in the inside, as well as a quadratic penalization of the density. Namely, we solve the problem 

min {  1/2 \int \frac{E^2}{2\mu} + \iint potentialV \mu + cCongestion \iint \mu^2   }
under the constraint d_t \mu + \nabla \cdot E = 0, and the values of mu at time 0 and 1 are fixed. 
"""


#********************************************************************************************************
# Package importation 
#********************************************************************************************************

# Clock
import time

# Mathematical functions
import numpy as np
import scipy.sparse as scsp
import scipy.sparse.linalg as scspl
from numpy import linalg as lin
from scipy.fftpack import * # FFT and DCT

from math import *



#********************************************************************************************************
# Main function 
#********************************************************************************************************

def geodesicBB1D(nO, nD, Nit, mub0, mub1, cCongestion = 0.0, potentialV = None, detailStudy = False, eps = 10**(-5) ) :
	
	"""
	Main function which is called to do the computation
	
	Inputs: 
	nO: number of discretization points of the source space [0,1]
	nD: number of discretization points per dimension of the target space
	Nit: number of ADMM iterations performed
	mub0, mub1: nD**2-arrays with temporal boundary conditions
	cCongestion: scale of the quadratic penalization of the density in the running cost 
	potentialV: nD**2-array thought as a function on D
	detailStudy: to decide wether we compute the objective functional every iteration or not (slower if marked true)  
	
	Outputs: 
	mu: nO*nD*nD array with the values of the density  
	E0, E1: nO*nD*nD arrays with the values of the momentum  
	objectiveValue: value of the Lagrangian along the iterations of the ADMM  
	primalResidual, dualResidual: values of the L^2 norms of the primal and dual residuals along the iterations of the ADMM
	"""

	startProgram = time.time() 
	
	print("Parameters ----------------------")
	print( "nO: " + str(nO) )
	print( "nD: " + str(nD) )
	print( "cCongestion: " + str(cCongestion) )
	print()
	

	#****************************************************************************************************
	# Domain Building
	#****************************************************************************************************

	# Domain Omega: centered grid 
	xOmegaC = np.linspace(0,1,nO)

	# Step Omega
	DeltaOmega = xOmegaC[1] - xOmegaC[0]

	# Domain Omega: staggered grid 
	xOmegaS = np.linspace(- DeltaOmega/2,1 + DeltaOmega/2 ,nO+1) 

	# Domain D (grid is "periodic") : centered grid 
	xDC = np.linspace(0,1 - 1/nD,nD)
	yDC = np.linspace(0,1 - 1/nD,nD)
	xGridDC, yGridDC = np.meshgrid(xDC, yDC, indexing = 'ij')

	# Step D
	DeltaD = xDC[1] - xDC[0]

	# Domain D: staggered grid 
	xDS = np.linspace(DeltaD/2,1 - DeltaD/2,nD)
	yDS = np.linspace(DeltaD/2,1 - DeltaD/2,nD)
	xGridDS, yGridDS = np.meshgrid(xDC, yDC, indexing = 'ij')

	# In D
	# The neighbors of the point i of centered are i-1 and i in staggered
	# The neighbors of the point i of staggered are i and i+1 in centered

	#***************************************************************************************************
	# Function building 
	#****************************************************************************************************
	
	# Lagrange multiplier associated to mu. Centered everywhere 
	mu = np.zeros((nO,nD,nD)) 

	# Momentum E, lagrange mutliplier. Centered everywhere. The two last components indicate to which dr_i phi^alpha it corresponds
	# First number is component in Omega, second is component in D   
	E0 = np.zeros((nO,nD,nD,2,2))
	E1 = np.zeros((nO,nD,nD,2,2))

	# Dual variable phi (phi^alpha staggered in alpha, centered everywhere else) 
	phi = np.zeros((nO+1,nD,nD))

	# Primal Variable A : A^alpha beta which corresponds to dr_beta phi^alpha. Same pattern as muTilde 
	A = np.zeros((nO,nD,nD))


	# Primal variable B, same pattern as E 
	# First number is component in Omega, second is component in D   
	B0 = np.zeros((nO,nD,nD,2,2))
	B1 = np.zeros((nO,nD,nD,2,2))

	# Lagrange multiplier associated to the congestion 
	lambdaC = np.zeros((nO,nD,nD))

	if potentialV is None : 
		potentialV = np.zeros((nD,nD))
	
	#****************************************************************************************************
	# Boundary values  
	#****************************************************************************************************

	# Normalization ----------------------------------------------------------------------------

	mub0 /= (np.sum(mub0)) 
	mub1 /= (np.sum(mub1)) 

	# Build the boundary term ----------------------------------------------------------------- 

	BT = np.zeros((nO+1,nD,nD))

	BT[0,:,:] = -mub0[:,:] / DeltaOmega

	BT[-1,:,:] = mub1[:,:] / DeltaOmega
		
	#****************************************************************************************************
	# Scalar product  	#***************************************************************************************************	
	def scalarProduct(a,b) : 
		return np.sum( np.multiply( a,b ) )
		
	#****************************************************************************************************
	# Differential, averaging and projection operators  
	#****************************************************************************************************

	# Derivate along Omega of a staggered function. Return a centered function --------------------------

	def gradOmega(input) : 
		output = (input[1:,:,:] - input[:-1,:,:]) / DeltaOmega
		return output
		
	# MINUS Adjoint of the two operator 
		
	def gradAOmega(input) : 

		inputSize = input.shape
		output = np.zeros( ( inputSize[0]+1, inputSize[1], inputSize[2] ) )
		
		output[1:-1,:,:] = (input[1:,:,:] - input[:-1,:,:]) / DeltaOmega
		output[0,:,:] = input[0,:,:] / DeltaOmega
		output[-1,:,:] = -input[-1,:,:] / DeltaOmega
		
		return output



	# Derivate along D of a staggered function. Return a centered function ---------------------------- 

	def gradD0(input) : 

		inputSize = input.shape
		output = np.zeros(inputSize)

		output[:,1:,:] = ( input[:,1:,:] - input[:,:-1,:] ) / DeltaD
		output[:,0,:] = ( input[:,0,:] - input[:,-1,:] ) / DeltaD
		
		return output
		
	def gradD1(input) : 

		inputSize = input.shape
		output = np.zeros(inputSize)

		output[:,:,1:] = ( input[:,:,1:] - input[:,:,:-1] ) / DeltaD
		output[:,:,0] = ( input[:,:,0] - input[:,:,-1] ) / DeltaD
		
		return output		
		
	# MINUS Adjoint of the two previous operators -- Same as derivative along D of a centered function, return a staggered one  

	def gradAD0(input) :

		inputSize = input.shape
		output = np.zeros(inputSize)

		output[:,:-1,:] = ( input[:,1:,:] - input[:,:-1,:] ) / DeltaD
		output[:,-1,:] = ( input[:,0,:] - input[:,-1,:] ) / DeltaD
		
		return output
		
	def gradAD1(input) : 

		inputSize = input.shape
		output = np.zeros(inputSize)

		output[:,:,:-1] = ( input[:,:,1:] - input[:,:,:-1] ) / DeltaD
		output[:,:,-1] = ( input[:,:,0] - input[:,:,-1] ) / DeltaD
		
		return output

	# Splitting operator and its adjoint ------------------------------------ 

	# The input has the same staggered pattern as grad_D phi 

	def splitting(input0, input1) :

		output0 = np.zeros((nO,nD,nD,2,2))
		output1 = np.zeros((nO,nD,nD,2,2))	
		
		# Output 0 
		
		output0[:,0,:,0,0] = input0[:-1,-1,:]
		output0[:,1:,:,0,0] = input0[:-1,:-1,:]
		
		output0[:,:,:,0,1] = input0[:-1,:,:]
		
		output0[:,0,:,1,0] = input0[1:,-1,:]
		output0[:,1:,:,1,0] = input0[1:,:-1,:]
		
		output0[:,:,:,1,1] = input0[1:,:,:]
		
		# Output 1
		
		output1[:,:,0,0,0] = input1[:-1,:,-1]
		output1[:,:,1:,0,0] = input1[:-1,:,:-1]
		
		output1[:,:,:,0,1] = input1[:-1,:,:]
		
		output1[:,:,0,1,0] = input1[1:,:,-1]
		output1[:,:,1:,1,0] = input1[1:,:,:-1]
		
		output1[:,:,:,1,1] = input1[1:,:,:]
		
		return output0, output1
		
		
	# Adjoint of the splitting operator. Take something which has the same staggered pattern as B, E and returns something which is like grad_D phi 

	def splittingA(input0, input1) :

		output0 = np.zeros( (nO+1,nD,nD) )
		output1 = np.zeros( (nO+1,nD,nD) )	
			
		# Output 0 
		
		output0[:-1,-1,:] += input0[:,0,:,0,0]
		output0[:-1,:-1,:] += input0[:,1:,:,0,0]
		
		output0[:-1,:,:] += input0[:,:,:,0,1]
		
		output0[1:,-1,:] += input0[:,0,:,1,0]
		output0[1:,:-1,:] += input0[:,1:,:,1,0]
		
		output0[1:,:,:] += input0[:,:,:,1,1]
		
		# Output 1 
		
		output1[:-1,:,-1] += input1[:,:,0,0,0]
		output1[:-1,:,:-1] += input1[:,:,1:,0,0]
		
		output1[:-1,:,:] += input1[:,:,:,0,1]
		
		output1[1:,:,-1] += input1[:,:,0,1,0]
		output1[1:,:,:-1] += input1[:,:,1:,1,0]
		
		output1[1:,:,:] += input1[:,:,:,1,1]	
		
		return output0, output1	
		
	# Returning the derivatives of phi -----------------------------------------------------------

	# Derivatives wrt Omega of phi 
	def derivativeOphi() :
		
		return gradOmega(phi)
		
	# Derivatives wrt D of phi. As phi is centered, we use the adjoint of the gradient   
		
	def derivativeDphi() :

		return gradAD0(phi), gradAD1(phi)	
	# Derivatives wrt D and splitting. Return an object which has the same staggered pattern as B and E 

	def derivativeSplittingDphi() : 

		output0 = np.zeros((nO,nD,nD,2,2))
		output1 = np.zeros((nO,nD,nD,2,2))
		
		
		dD0phi, dD1phi = derivativeDphi() 
		
		output0, output1 = splitting(dD0phi, dD1phi)
		
		return output0, output1
		
	#****************************************************************************************************
	# Laplace Matrix and preconditionner for its inverse 
	#****************************************************************************************************
			
	# Build the Laplace operator ------------------------------------- 

	auxCrap0 = np.zeros((nO+1,nD,nD))
	auxCrap1 = np.zeros((nO+1,nD,nD))

	def LaplaceFunction(input) : 

		inputShaped = input.reshape((nO+1,nD,nD))

		output = np.zeros( (nO+1,nD,nD) )
		
		# Derivatives in Omega
		output += gradAOmega( gradOmega( inputShaped ) )

		# Derivatives and splitting in D 
		
		aux0, aux1 = splitting(  gradAD0(inputShaped), gradAD1(inputShaped))
		dD0, dD1 = splittingA(  aux0, aux1  ) 
		
		# Update output 
		
		output += gradD0( dD0 )
		output += gradD1( dD1 )
		
		return  output.reshape((nO+1)*nD*nD) + eps*input
		

		
	LaplaceOp = scspl.LinearOperator(((nO+1)*nD*nD,(nO+1)*nD*nD), matvec=LaplaceFunction)	

	# Build the preconditionner for the Laplace matrix using FFT: here we do not take into account S*S, where S is the splitting operator, it is why it is only a preconditionner and not the true inverse
		
	# We build the diagonal coefficients in the Fourier basis -----------------------------------------

	# Beware of the fact that there is a factor 4 in front of the derivatives in D because of the multiplicity of E

	LaplaceDiagInv = np.zeros( (nO+1,nD,nD) )

	for alpha in range(nO+1) : 
		for i in range(nD) : 
			for j in range(nD) : 
					
				toInv = 0.0 
					
				# Derivatives in Omega 
				toInv += 2 * (1 - cos(alpha*pi / nO)) / (DeltaOmega**2)
			
				# Derivatives in D 
				toInv += 8* (1 - cos(2*pi*i/nD)) / (DeltaD**2)
				toInv += 8* (1 - cos(2*pi*j/nD)) / (DeltaD**2)
					
				if abs(toInv) <= 10**(-10) :
					LaplaceDiagInv[alpha,i,j] = 0.0			
				else : 
					LaplaceDiagInv[alpha, i,j] = - 1 / ( toInv + eps)
							
		
	# Compute the multiplicative constant for the operator idct( dct )

	AuxFFT = np.random.rand(nO)
	ImAuxFFT = dct( AuxFFT, type = 1) 
	InvAuxFFT = idct( ImAuxFFT, type = 1)

	ConstantFFT = AuxFFT[0] / InvAuxFFT[0]		

	# Then we build the preconditionners as functions 

	def precondFunction(input) : 
			
		inputShaped = input.reshape((nO+1,nD,nD)) 
			
		# Applying FFT 
		
		input_FFT = dct(  fft( fft( inputShaped, axis = 2 ), axis = 1) , type = 1, axis = 0 ) 
			
		# Multiplication by the diagonal matrix 
		
		solution_FFT = np.multiply( LaplaceDiagInv, input_FFT ) 	

		# Inverse transformation 
		
		solution = ConstantFFT * idct(  ifft( ifft( solution_FFT, axis = 2 ), axis = 1) , type = 1, axis = 0 )

		# Storage of the results 
		
		output = solution.real
		
		return output.reshape((nO+1)*nD*nD)
		
	# And Finally we transform them as operators 

	precondOp = scspl.LinearOperator(((nO+1)*nD*nD,(nO+1)*nD*nD), matvec=precondFunction)
		
	#****************************************************************************************************
	# Objective functional 
	#****************************************************************************************************
			
	def objectiveFunctional() : 
		
		output = 0.0 
		
		# Boundary term 
		output += scalarProduct( phi, BT ) * DeltaOmega
		
		# Computing the derivatives of phi and split them in D 
		dOphi = derivativeOphi()
		dSD0phi, dSD1phi = derivativeSplittingDphi()

		# Lagrange multiplier mu  
		output += scalarProduct( A + lambdaC - dOphi + potentialV, mu ) * DeltaOmega
		
		# Lagrange multiplier E. 
		output += scalarProduct( B0 - dSD0phi, E0 ) * DeltaOmega
		output += scalarProduct( B1 - dSD1phi, E1 ) * DeltaOmega
		
		# Penalization of congestion 
		if abs(  cCongestion ) >= 10**(-8) :
			output -= 1/(2.*cCongestion) * scalarProduct( lambdaC, lambdaC ) * DeltaOmega * DeltaD**2
				
		# Penalty in A, phi 
		output -= r/2 * scalarProduct( A + lambdaC + potentialV - dOphi, A + lambdaC + potentialV - dOphi ) * DeltaOmega * DeltaD**2 
		
		# Penalty in B, phi. 
		output -= r/2* scalarProduct( B0 - dSD0phi, B0 - dSD0phi ) * DeltaOmega * DeltaD**2 
		output -= r/2* scalarProduct( B1 - dSD1phi, B1 - dSD1phi ) * DeltaOmega * DeltaD**2 
		
		return output
		
	#****************************************************************************************************
	# Algorithm iteration 
	#****************************************************************************************************

	# Value of the augmentation parameter (updated during the ADMM iterations)
	r = 1. 
	
	# Initialize the array which will contain the values of the objective functional 
	if detailStudy:
		objectiveValue = np.zeros( 3*Nit )
	else : 
		objectiveValue = np.zeros( (Nit // 10) )

	# Residuals 
	primalResidual = np.zeros(Nit)
	dualResidual = np.zeros(Nit)

	# Main Loop

	for counterMain in range(Nit) :
		
		print( 30*"-" +  " Iteration " + str(counterMain + 1) + " " + 30*"-"  )
		
		if detailStudy:
			objectiveValue[3*counterMain] = objectiveFunctional()
		elif (counterMain % 10) == 0 : 
			objectiveValue[ counterMain // 10 ] = objectiveFunctional()
		
		# Laplace problem -----------------------------------------------------------------------------

		startLaplace = time.time()
		
		# Build the RHS 
		
		RHS = np.zeros(  (nO+1,nD,nD) )
		
		RHS -= BT * DeltaOmega 
		
		RHS -=  gradAOmega(  mu   ) * DeltaOmega
		
		RHS += r*gradAOmega(  A + lambdaC + potentialV   ) * DeltaOmega * DeltaD**2
		
		# Take the splitting adjoint of both E and B 
		ES0, ES1 = splittingA(  E0, E1 )
		BS0, BS1 = splittingA(  B0, B1 )
		
		RHS -= gradD0(  ES0  ) * DeltaOmega
		RHS -= gradD1(  ES1  ) * DeltaOmega
			
		RHS += r*gradD0(  BS0  ) * DeltaOmega * DeltaD**2
		RHS += r*gradD1(  BS1  ) * DeltaOmega * DeltaD**2
			
		# Solve the system 
		
		solution, res = scspl.cg(LaplaceOp,RHS.reshape(((nO+1)*nD*nD)), M = precondOp, maxiter = 50 )
		
		# print("Resolution of the linear system: " + str(res))
		
		phi = solution.reshape((nO+1,nD,nD)) / ( r* DeltaOmega * DeltaD**2 )
		
		endLaplace = time.time()
		print( "Solving the Laplace system: " + str( round( endLaplace - startLaplace, 2) ) + "s." )
		
		if detailStudy:
			objectiveValue[3*counterMain + 1] = objectiveFunctional()
		
		# Projection over a convex set ---------------------------------------------------------
		# It projects on the set Tr(A) + 1/2 |B|^2 <= 0. We reduce to a 1D projection, then use a Newton method with a fixed number of iteration. 
		
		startProj = time.time()
			
		# Computing the derivatives of phi and split them in D 
		dOphi = derivativeOphi()
		dSD0phi, dSD1phi = derivativeSplittingDphi()
		
		# Compute what needs to be projected
		
		# On A 
		aArray =  dOphi - potentialV + mu /( r * DeltaD**2) 
		
		# On B
		toProjectB0 = dSD0phi + E0 /( r * DeltaD**2)
		toProjectB1 = dSD1phi + E1 /( r * DeltaD**2)
			
		bSquaredArray = np.sum( np.square(toProjectB0) + np.square(toProjectB1) , axis = (-1,-2) ) / 8  
		
		# Compute the array discriminating between the values already on the convex and the others 
		# Value of the objective functional. For the points not in the convex, we want it to vanish.
		projObjective = aArray + bSquaredArray 
		# projDiscriminating is 1 is the point needs to be projected, 0 if it is already in the convex 
		projDiscriminating = np.greater( projObjective, 10**(-16) * np.ones( (nO,nD,nD) ) )
		projDiscriminating = projDiscriminating.astype(int)
		projDiscriminating = projDiscriminating.astype(float)

		# Newton method iteration 
		
		# Value of the Lagrange multiplier. Initialized at 0, not updated if already in the convex set 
		xProj = np.zeros((nO,nD,nD))
		
		for counterProj in range(20) : 
			# Objective functional
			projObjective = aArray + 4*(1. + cCongestion * r)* xProj + np.divide(bSquaredArray,  np.square( 1 - xProj ) )
			# Derivative of the ojective functional 
			dProjObjective = 4*(1. + cCongestion * r) - 2 * np.divide( bSquaredArray, np.power( xProj - 1 , 3) ) 
			# Update of xProj 
			xProj -= np.divide(np.multiply( projDiscriminating, projObjective ), dProjObjective)
				
		# Update of A and B as a result 
		
		A = aArray + 4*(1. + cCongestion * r)* xProj
		
		# Update lambdaC
		lambdaC = - 4* cCongestion * r * xProj
		
		# Rescale xProj so as it has the same dimension as B and E 
		xProj = np.kron(  xProj.reshape((nO*nD*nD)) , np.ones(4) ).reshape((nO,nD,nD,2,2))	
		
		B0 = np.divide( toProjectB0, (1 - xProj) )
		B1 = np.divide( toProjectB1, (1 - xProj))
		
			
		# Print the info 
		
		endProj = time.time()
		print( "Pointwise projection: " + str( round( endProj - startProj, 2) ) + "s." )
		
		if detailStudy:
			objectiveValue[3*counterMain + 2] = objectiveFunctional()	
			
		# Gradient descent in (E,muTilde), i.e. in the dual ----------------------------------------------- 
			
		# No need to recompute the derivatives of phi  
		
		# Update for muTilde -- no need to update the cross terms, they vanish 
		mu -= r * ( A + lambdaC + potentialV - dOphi ) * DeltaD**2
		
		# Update for E 
		E0 -= r * ( B0 - dSD0phi ) * DeltaD**2
		E1 -= r * ( B1 - dSD1phi ) * DeltaD**2
		
		# Compute the residuals ------------------------------------------------------------------ 
		
		# For the primal residual, just sum what was the update in the dual 
		primalResidual[counterMain] = sqrt(DeltaOmega) * DeltaD * lin.norm(  np.array(  [ lin.norm(A + lambdaC + potentialV - dOphi), lin.norm(  B0 - dSD0phi ), lin.norm( B1 - dSD1phi )] )) 
		
		
		# For the residual, take the RHS of the Laplace system and conserve only BT and the dual variables mu, E 
		
		dualResidualAux = np.zeros((nO+1,nD,nD))
		
		dualResidualAux -= BT 
		
		dualResidualAux -=  gradAOmega(  mu  )  
		
		# Take the splitting adjoint of both E and B 
		ES0, ES1 = splittingA(  E0, E1 )	
		
		dualResidualAux -= gradD0(  ES0  )
		dualResidualAux -= gradD1(  ES1  )
		
		dualResidual[counterMain] = r*sqrt(DeltaOmega)*lin.norm(dualResidualAux )
		
		# Update the parameter r -----------------------------------------------------------------
		
		# cf. Boyd et al. for an explanantion of the rule
		
		if primalResidual[counterMain] >= 10 * dualResidual[counterMain] : 
			r *= 2 
		elif 10* primalResidual[counterMain] <=  dualResidual[counterMain] : 
			r /= 2
		
		# Printing some results ------------------------------------------------------------------
		
		if detailStudy:
		
			print("Maximizing in phi, should go up: " + str( objectiveValue[3*counterMain + 1] - objectiveValue[3*counterMain]   )  )
			print("Maximizing in A,B, should go up: " + str(  objectiveValue[3*counterMain + 2] - objectiveValue[3*counterMain + 1] ) )
			if counterMain >= 1 : 
				print("Dual update: should go down: " + str( objectiveValue[3*counterMain] - objectiveValue[3*counterMain-1]  )  )
			
		print("Values of phi0:")
		print(np.max(phi))
		print(np.min(phi))
		
		print("Values of A")
		print(np.max(A))
		print(np.min(A))
		
		print("Values of mu")
		print(np.max(mu))
		print(np.min(mu))
		
		print("Values of E0")
		print(np.max(E0))
		print(np.min(E0))

		print("r")
		print(r)
		
	#****************************************************************************************************
	# End of the program, printing  and returning the results 	#****************************************************************************************************	
	print()
	print( 30*"-" +  " End of the ADMM iterations " + 30*"-"  )
	
	# Integral of the density	
	intMu = np.sum( mu, axis=(-1,-2) )

	print("Minimal and maximal value of integral of the density")
	print(np.min(intMu))
	print(np.max(intMu))


	print("Maximal and minimal value of the density")
	print(np.min(mu) / DeltaD**2)
	print(np.max(mu) / DeltaD**2)

	print("Discrepancy between lambdaC and mu")
	print(  np.max( lambdaC - mu * cCongestion / (DeltaD**2)  ) )

	print("Final value of the augmentation paramter")
	print(r)

	print("Final value of the objective functional")
	print( objectiveValue[-1] )

	endProgramm = time.time()

	print( "Total time taken by the program: " + str( round( endProgramm - startProgram, 2) ) + "s." )
	
	return mu, E0, E1, objectiveValue, primalResidual, dualResidual
	





