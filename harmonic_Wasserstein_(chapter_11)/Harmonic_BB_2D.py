""" 
Solution of the Dirichlet problem via a Benamou Brenier method

Starting domain Omega : Unit square of R^2 
Target domain D : Torus of 2 dimension

We use ALG2 on the dual problem
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

def harmonicBB2D(nO, nD, Nit, mub0, mub1, detailStudy = False, eps = 10**(-5) ) :

	"""
	Main function which is called to do the computation
	
	Inputs: 
	nO: number of discretization points per dimension of the source space [0,1]**2
	nD: number of discretization points per dimension of the target space
	Nit: number of ADMM iterations performed
	mub0, mub1: arrays with the boundary conditions. mub0 corresponds to the vertical boundaries of the square (absciss = cst), while mub1 corresponds to the horizontal ones (absciss = cst). mub0 is a (nO+1)*nO*nD*nD array, while mub1 is a nO*(nO+1)*nD*nD  
	detailStudy: to decide wether we compute the objective functional every iteration or not (slower if marked true)  
	
	Outputs: 
	mu: nO*nD*nD array with the values of the density  
	E00, E01, E10, E11: arrays with the values of the momentum  
	objectiveValue: value of the Lagrangian along the iterations of the ADMM  
	primalResidual, dualResidual: values of the L^2 norms of the primal and dual residuals along the iterations of the ADMM
	"""

	startProgram = time.time() 
	
	print("Parameters ----------------------")
	print( "nO: " + str(nO) )
	print( "nD: " + str(nD) )
	print()

	#****************************************************************************************************
	# Domain Building
	#****************************************************************************************************

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

	# Lagrange multiplier associated to mu (diagonal coefficients are centered, anti-diagonal are staggered in Omega)
	# muTilde alpha beta corresponds to dr_beta phi^alpha 
	muTilde00 = np.zeros((nO,nO,nD,nD)) 
	muTilde01 = np.zeros((nO+1,nO-1,nD,nD)) 
	muTilde10 = np.zeros((nO-1,nO+1,nD,nD)) 
	muTilde11 = np.zeros((nO,nO,nD,nD)) 


	# Momentum E, lagrange mutliplier. Centered everywhere. The two last components indicate to which dr_i phi^alpha it corresponds
	# First number is component in Omega, second is component in D   
	E00 = np.zeros((nO,nO,nD,nD,2,2))
	E01 = np.zeros((nO,nO,nD,nD,2,2))
	E10 = np.zeros((nO,nO,nD,nD,2,2))
	E11 = np.zeros((nO,nO,nD,nD,2,2))

	# Dual variable phi (phi^alpha staggered in alpha, centered everywhere else) 
	phi0 = np.zeros((nO+1,nO,nD,nD))
	phi1 = np.zeros((nO,nO+1,nD,nD))

	# Primal Variable A : A^alpha beta which corresponds to dr_beta phi^alpha. Same pattern as muTilde 
	A00 = np.zeros((nO,nO,nD,nD))
	A01 = np.zeros((nO+1,nO-1,nD,nD))
	A10 = np.zeros((nO-1,nO+1,nD,nD))
	A11 = np.zeros((nO,nO,nD,nD))


	# Primal variable B, same pattern as E 
	# First number is component in Omega, second is component in D   
	B00 = np.zeros((nO,nO,nD,nD,2,2))
	B01 = np.zeros((nO,nO,nD,nD,2,2))
	B10 = np.zeros((nO,nO,nD,nD,2,2))
	B11 = np.zeros((nO,nO,nD,nD,2,2))


	#****************************************************************************************************
	# Boundary values  
	#****************************************************************************************************

				
	# Normalization ----------------------------------------------------------------------------

	mub0[0,:,:,:] = np.divide( mub0[0,:,:,:], (np.kron( (np.sum(mub0[0,:,:,:], axis = (-1,-2))).reshape((nO,1)) , np.ones((nD,nD)) )).reshape( (nO,nD,nD) ) )	
	mub0[-1,:,:,:] = np.divide( mub0[-1,:,:,:], (np.kron( (np.sum(mub0[-1,:,:,:], axis = (-1,-2))).reshape((nO,1)) , np.ones((nD,nD)) )).reshape( (nO,nD,nD) ) )
	mub1[:,0,:,:] = np.divide( mub1[:,0,:,:], (np.kron( (np.sum(mub1[:,0,:,:], axis = (-1,-2))).reshape((nO,1)) , np.ones((nD,nD)) )).reshape( (nO,nD,nD) ) )
	mub1[:,-1,:,:] = np.divide( mub1[:,- 1,:,:], (np.kron( (np.sum(mub1[:,-1,:,:], axis = (-1,-2))).reshape((nO,1)) , np.ones((nD,nD)) )).reshape( (nO,nD,nD) ) )

	# Build the boundary term ----------------------------------------------------------------- 

	BT0 = np.zeros((nO+1,nO,nD,nD))
	BT1 = np.zeros((nO,nO+1,nD,nD))

	BT0[0,:,:,:] = -mub0[0,:,:,:] / DeltaOmega

	BT0[-1,:,:,:] = mub0[-1,:,:,:] / DeltaOmega
		
	BT1[:,0,:,:] = -mub1[:,0,:,:] / DeltaOmega

	BT1[:,-1,:,:] = mub1[:,-1,:,:] / DeltaOmega

	#****************************************************************************************************# Scalar product  	#****************************************************************************************************	
		
	def scalarProduct(a,b) : 
		return np.sum( np.multiply( a,b ) )
		
	#****************************************************************************************************
	# Differential, averaging and projection operators  
	#****************************************************************************************************

	# Derivate along Omega of a staggered function. Return a centered function --------------------------

	def gradOmega0(input) : 
		output = (input[1:,:,:,:] - input[:-1,:,:,:]) / DeltaOmega
		return output
		
	def gradOmega1(input) : 
		output = (input[:,1:,:,:] - input[:,:-1,:,:]) / DeltaOmega
		return output
		
	# MINUS Adjoint of the two previous operators 
		
	def gradAOmega0(input) : 

		inputSize = input.shape
		output = np.zeros( ( inputSize[0]+1, inputSize[1], inputSize[2], inputSize[3] ) )
		
		output[1:-1,:,:,:] = (input[1:,:,:,:] - input[:-1,:,:,:]) / DeltaOmega
		output[0,:,:,:] = input[0,:,:,:] / DeltaOmega
		output[-1,:,:,:] = -input[-1,:,:,:] / DeltaOmega
		
		return output

	def gradAOmega1(input) : 

		inputSize = input.shape
		output = np.zeros( ( inputSize[0], inputSize[1]+1, inputSize[2], inputSize[3] ) )
		
		output[:,1:-1,:,:] = (input[:,1:,:,:] - input[:,:-1,:,:]) / DeltaOmega
		output[:,0,:,:] = input[:,0,:,:] / DeltaOmega
		output[:,-1,:,:] = -input[:,-1,:,:] / DeltaOmega
		
		return output	

	# Derivate along D of a staggered function. Return a centered function ---------------------------- 

	def gradD0(input) : 

		inputSize = input.shape
		output = np.zeros(inputSize)

		output[:,:,1:,:] = ( input[:,:,1:,:] - input[:,:,:-1,:] ) / DeltaD
		output[:,:,0,:] = ( input[:,:,0,:] - input[:,:,-1,:] ) / DeltaD
		
		return output
		
	def gradD1(input) : 

		inputSize = input.shape
		output = np.zeros(inputSize)

		output[:,:,:,1:] = ( input[:,:,:,1:] - input[:,:,:,:-1] ) / DeltaD
		output[:,:,:,0] = ( input[:,:,:,0] - input[:,:,:,-1] ) / DeltaD
		
		return output		
		
	# MINUS Adjoint of the two previous operators -- Same as derivative along D of a centered function, return a staggered one  

	def gradAD0(input) :

		inputSize = input.shape
		output = np.zeros(inputSize)

		output[:,:,:-1,:] = ( input[:,:,1:,:] - input[:,:,:-1,:] ) / DeltaD
		output[:,:,-1,:] = ( input[:,:,0,:] - input[:,:,-1,:] ) / DeltaD
		
		return output
		
	def gradAD1(input) : 

		inputSize = input.shape
		output = np.zeros(inputSize)

		output[:,:,:,:-1] = ( input[:,:,:,1:] - input[:,:,:,:-1] ) / DeltaD
		output[:,:,:,-1] = ( input[:,:,:,0] - input[:,:,:,-1] ) / DeltaD
		
		return output

	# Splitting operator and its adjoint ------------------------------------ 

	# The input has the same staggered pattern as grad_D phi 

	def splitting(input00, input01, input10, input11) :

		output00 = np.zeros((nO,nO,nD,nD,2,2))
		output01 = np.zeros((nO,nO,nD,nD,2,2))
		output10 = np.zeros((nO,nO,nD,nD,2,2))
		output11 = np.zeros((nO,nO,nD,nD,2,2))
		
		# Output 00 
		
		output00[:,:,0,:,0,0] = input00[:-1,:,-1,:]
		output00[:,:,1:,:,0,0] = input00[:-1,:,:-1,:]
		
		output00[:,:,:,:,0,1] = input00[:-1,:,:,:]
		
		output00[:,:,0,:,1,0] = input00[1:,:,-1,:]
		output00[:,:,1:,:,1,0] = input00[1:,:,:-1,:]
		
		output00[:,:,:,:,1,1] = input00[1:,:,:,:]
		
		# Output 01
		
		output01[:,:,:,0,0,0] = input01[:-1,:,:,-1]
		output01[:,:,:,1:,0,0] = input01[:-1,:,:,:-1]
		
		output01[:,:,:,:,0,1] = input01[:-1,:,:,:]
		
		output01[:,:,:,0,1,0] = input01[1:,:,:,-1]
		output01[:,:,:,1:,1,0] = input01[1:,:,:,:-1]
		
		output01[:,:,:,:,1,1] = input01[1:,:,:,:]
		
		# Output 10 
		
		output10[:,:,0,:,0,0] = input10[:,:-1,-1,:]
		output10[:,:,1:,:,0,0] = input10[:,:-1,:-1,:]
		
		output10[:,:,:,:,0,1] = input10[:,:-1,:,:]
		
		output10[:,:,0,:,1,0] = input10[:,1:,-1,:]
		output10[:,:,1:,:,1,0] = input10[:,1:,:-1,:]
		
		output10[:,:,:,:,1,1] = input10[:,1:,:,:]
		
		# Output 11
		
		output11[:,:,:,0,0,0] = input11[:,:-1,:,-1]
		output11[:,:,:,1:,0,0] = input11[:,:-1,:,:-1]
		
		output11[:,:,:,:,0,1] = input11[:,:-1,:,:]
		
		output11[:,:,:,0,1,0] = input11[:,1:,:,-1]
		output11[:,:,:,1:,1,0] = input11[:,1:,:,:-1]
		
		output11[:,:,:,:,1,1] = input11[:,1:,:,:]
		
		return output00, output01, output10, output11
		
		
	# Adjoint of the splitting operator. Take something which has the same staggered pattern as B, E and returns something which is like grad_D phi 

	def splittingA(input00, input01, input10, input11) :

		output00 = np.zeros( (nO+1,nO,nD,nD) )
		output01 = np.zeros( (nO+1,nO,nD,nD) )	
		output10 = np.zeros( (nO,nO+1,nD,nD) )
		output11 = np.zeros( (nO,nO+1,nD,nD) )
		
		# Output 00 
		
		output00[:-1,:,-1,:] += input00[:,:,0,:,0,0]
		output00[:-1,:,:-1,:] += input00[:,:,1:,:,0,0]
		
		output00[:-1,:,:,:] += input00[:,:,:,:,0,1]
		
		output00[1:,:,-1,:] += input00[:,:,0,:,1,0]
		output00[1:,:,:-1,:] += input00[:,:,1:,:,1,0]
		
		output00[1:,:,:,:] += input00[:,:,:,:,1,1]
		
		# Output 01 
		
		output01[:-1,:,:,-1] += input01[:,:,:,0,0,0]
		output01[:-1,:,:,:-1] += input01[:,:,:,1:,0,0]
		
		output01[:-1,:,:,:] += input01[:,:,:,:,0,1]
		
		output01[1:,:,:,-1] += input01[:,:,:,0,1,0]
		output01[1:,:,:,:-1] += input01[:,:,:,1:,1,0]
		
		output01[1:,:,:,:] += input01[:,:,:,:,1,1]
		
		# Output 10 
		
		output10[:,:-1,-1,:] += input10[:,:,0,:,0,0]
		output10[:,:-1,:-1,:] += input10[:,:,1:,:,0,0]
		
		output10[:,:-1,:,:] += input10[:,:,:,:,0,1]
		
		output10[:,1:,-1,:] += input10[:,:,0,:,1,0]
		output10[:,1:,:-1,:] += input10[:,:,1:,:,1,0]
		
		output10[:,1:,:,:] += input10[:,:,:,:,1,1]
		
		# Output 11 
		
		output11[:,:-1,:,-1] += input11[:,:,:,0,0,0]
		output11[:,:-1,:,:-1] += input11[:,:,:,1:,0,0]
		
		output11[:,:-1,:,:] += input11[:,:,:,:,0,1]
		
		output11[:,1:,:,-1] += input11[:,:,:,0,1,0]
		output11[:,1:,:,:-1] += input11[:,:,:,1:,1,0]
		
		output11[:,1:,:,:] += input11[:,:,:,:,1,1]
		
		return output00, output01, output10, output11	
		
	# Returning the derivatives of phi -----------------------------------------------------------

	# Derivatives wrt Omega of phi 
	def derivativeOphi() :
		
		return gradOmega0(phi0), gradOmega1(phi0), gradOmega0(phi1), gradOmega1(phi1) 
		
	# Derivatives wrt D of phi. As phi is centered, we use the adjoint of the gradient   
		
	def derivativeDphi() :

		return gradAD0(phi0), gradAD1(phi0), gradAD0(phi1), gradAD1(phi1) 
		
	# Derivatives wrt D and splitting. Return an object which has the same staggered pattern as B and E 

	def derivativeSplittingDphi() : 

		output00 = np.zeros((nO,nO,nD,nD,2,2))
		output01 = np.zeros((nO,nO,nD,nD,2,2))
		output10 = np.zeros((nO,nO,nD,nD,2,2))
		output11 = np.zeros((nO,nO,nD,nD,2,2))
		
		dD0phi0, dD1phi0, dD0phi1, dD1phi1 = derivativeDphi() 
		
		output00, output01, output10, output11 = splitting(dD0phi0, dD1phi0, dD0phi1, dD1phi1)
		
		return output00, output01, output10, output11
		

	#****************************************************************************************************
	# Laplace Matrix and projection on it 
	#****************************************************************************************************
			
	# Build the Laplace operator ------------------------------------- 

	auxCrap00 = np.zeros((nO+1,nO,nD,nD))
	auxCrap01 = np.zeros((nO+1,nO,nD,nD))
	auxCrap10 = np.zeros((nO,nO+1,nD,nD))
	auxCrap11 = np.zeros((nO,nO+1,nD,nD))

	def LaplaceFunction0(input) : 

		inputShaped = input.reshape((nO+1,nO,nD,nD))

		output = np.zeros( (nO+1,nO,nD,nD) )
		
		# Derivatives in Omega
		output += gradAOmega0( gradOmega0( inputShaped ) )
		output += gradAOmega1( gradOmega1( inputShaped ) )

		# Derivatives and splitting in D 
		
		aux00, aux01, aux10, aux11 = splitting(  gradAD0(inputShaped), gradAD1(inputShaped), auxCrap10, auxCrap11)
		dD0, dD1, auxCrap0, auxCrap1 = splittingA(  aux00, aux01, aux10, aux11  ) 
		
		# Update output 
		
		output += gradD0( dD0 )
		output += gradD1( dD1 )
		
		return  output.reshape((nO+1)*nO*nD*nD) + eps*input
		
	def LaplaceFunction1(input) : 

		inputShaped = input.reshape((nO,nO+1,nD,nD))

		output = np.zeros( (nO,nO+1,nD,nD) )
		
		# Derivatives in Omega
		output += gradAOmega0( gradOmega0( inputShaped ) )
		output += gradAOmega1( gradOmega1( inputShaped ) )

		# Derivatives and splitting in D 
		
		aux00, aux01, aux10, aux11 = splitting(  auxCrap00, auxCrap01, gradAD0(inputShaped), gradAD1(inputShaped) )
		auxCrap0, auxCrap1, dD0, dD1  = splittingA(  aux00, aux01, aux10, aux11  ) 
		
		# Update output 
		
		output += gradD0( dD0 )
		output += gradD1( dD1 )
		
		return output.reshape(nO*(nO+1)*nD*nD) + eps*input
		
	LaplaceOp0 = scspl.LinearOperator(((nO+1)*nO*nD*nD,(nO+1)*nO*nD*nD), matvec=LaplaceFunction0)	
	LaplaceOp1 = scspl.LinearOperator((nO*(nO+1)*nD*nD,nO*(nO+1)*nD*nD), matvec=LaplaceFunction1)

	# Build the preconditionner for the Laplace matrix using FFT: here we do not take into account S*S, where S is the splitting operator, it is why it is only a preconditionner and not the true inverse
		
	# We build the diagonal coefficients in the Fourier basis  ---------------------------------------

	# Beware of the fact that there is a factor 4 in front of the derivatives in D because of the multiplicity of E

	LaplaceDiagInv0 = np.zeros( (nO+1,nO,nD,nD) )
	LaplaceDiagInv1 = np.zeros( (nO,nO+1,nD,nD) )


	for alpha in range(nO+1) : 
		for beta in range(nO) : 
			for i in range(nD) : 
				for j in range(nD) : 
					
					toInv = 0.0 
					
					# Derivatives in Omega 
					toInv += 2 * (1 - cos(alpha*pi / nO)) / (DeltaOmega**2)
					toInv += 2 * (1 - cos(beta*pi / (nO-1) ) ) / (DeltaOmega**2)
		
					# Derivatives in D 
					toInv += 8* (1 - cos(2*pi*i/nD)) / (DeltaD**2)
					toInv += 8* (1 - cos(2*pi*j/nD)) / (DeltaD**2)
					
					if abs(toInv) <= 10**(-10) :
						LaplaceDiagInv0[alpha,beta,i,j] = 0.0
						LaplaceDiagInv1[beta,alpha,i,j] = 0.0					
					else : 
						LaplaceDiagInv0[alpha, beta, i,j] = - 1 / ( toInv + eps)
						LaplaceDiagInv1[beta, alpha, i,j] = - 1 / ( toInv + eps)
						
	# Compute the multiplicative constant for the operator idct( dct )

	AuxFFT = np.random.rand(nO+1,nO+1)
	ImAuxFFT = dct( dct( AuxFFT, type = 1, axis = 1), type = 1, axis = 0) 
	InvAuxFFT = idct( idct( ImAuxFFT, type = 1, axis = 1), type = 1, axis = 0)

	ConstantFFT = AuxFFT[0,0] / InvAuxFFT[0,0]		

	# Then we build the preconditionners as functions 

	def precondFunction0(input) : 
			
		inputShaped = input.reshape((nO+1,nO,nD,nD)) 
			
		# Applying FFT 
		
		input_FFT = dct( dct(  fft( fft( inputShaped, axis = 3 ), axis = 2) , type = 1, axis = 1 ), type = 1, axis = 0) 
			
		# Multiplication by the diagonal matrix 
		
		solution_FFT = np.multiply( LaplaceDiagInv0, input_FFT ) 	

		# Inverse transformation 
		
		solution = ConstantFFT * idct( idct(  ifft( ifft( solution_FFT, axis = 3 ), axis = 2) , type = 1, axis = 1 ), type = 1, axis = 0) 

		# Storage of the results 
		
		output = solution.real
		
		return output.reshape((nO+1)*nO*nD*nD)
		
	def precondFunction1(input) : 
			
		inputShaped = input.reshape((nO,nO+1,nD,nD)) 
			
		# Applying FFT 
		
		input_FFT = dct( dct(  fft( fft( inputShaped, axis = 3 ), axis = 2) , type = 1, axis = 1 ), type = 1, axis = 0) 
			
		# Multiplication by the diagonal matrix 
		
		solution_FFT = np.multiply( LaplaceDiagInv1, input_FFT ) 	

		# Inverse transformation 
		
		solution = ConstantFFT * idct( idct(  ifft( ifft( solution_FFT, axis = 3 ), axis = 2) , type = 1, axis = 1 ), type = 1, axis = 0) 

		# Storage of the results 
		
		output = solution.real
		
		return output.reshape(nO*(nO+1)*nD*nD)
		
	# And Finally we transform them as operators 

	precondOp0 = scspl.LinearOperator(((nO+1)*nO*nD*nD,(nO+1)*nO*nD*nD), matvec=precondFunction0)	
	precondOp1 = scspl.LinearOperator((nO*(nO+1)*nD*nD,nO*(nO+1)*nD*nD), matvec=precondFunction1)
		
	#****************************************************************************************************
	# Objective functional 
	#****************************************************************************************************
			
	def objectiveFunctional() : 
		
		output = 0.0 
		
		# Boundary term 
		output += scalarProduct( phi0, BT0 ) * DeltaOmega**2
		output += scalarProduct( phi1, BT1 ) * DeltaOmega**2
		
		# Computing the derivatives of phi and split them in D 
		dO0phi0, dO1phi0, dO0phi1, dO1phi1 = derivativeOphi()
		dSD0phi0, dSD1phi0, dSD0phi1, dSD1phi1 = derivativeSplittingDphi()

		# Lagrange multiplier mu  
		output += scalarProduct( A00 - dO0phi0, muTilde00 ) * DeltaOmega**2
		output += scalarProduct( A01 - dO1phi0, muTilde01 ) * DeltaOmega**2
		output += scalarProduct( A10 - dO0phi1, muTilde10 ) * DeltaOmega**2
		output += scalarProduct( A11 - dO1phi1, muTilde11 ) * DeltaOmega**2
		
		# Lagrange multiplier E. 
		output += scalarProduct( B00 - dSD0phi0, E00 ) * DeltaOmega**2
		output += scalarProduct( B01 - dSD1phi0, E01 ) * DeltaOmega**2
		output += scalarProduct( B10 - dSD0phi1, E10 ) * DeltaOmega**2
		output += scalarProduct( B11 - dSD1phi1, E11 ) * DeltaOmega**2
		
		# Penalty in A, phi 
		output -= r/2 * scalarProduct( A00 - dO0phi0, A00 - dO0phi0 ) * DeltaOmega**2 * DeltaD**2
		output -= r/2 * scalarProduct( A01 - dO1phi0, A01 - dO1phi0 ) * DeltaOmega**2 * DeltaD**2
		output -= r/2 * scalarProduct( A10 - dO0phi1, A10 - dO0phi1 ) * DeltaOmega**2 * DeltaD**2
		output -= r/2 * scalarProduct( A11 - dO1phi1, A11 - dO1phi1 ) * DeltaOmega**2 * DeltaD**2
		
		# Penalty in B, phi. 
		output -= r/2* scalarProduct( B00 - dSD0phi0, B00 - dSD0phi0 ) * DeltaOmega**2 * DeltaD**2
		output -= r/2* scalarProduct( B01 - dSD1phi0, B01 - dSD1phi0 ) * DeltaOmega**2 * DeltaD**2
		output -= r/2* scalarProduct( B10 - dSD0phi1, B10 - dSD0phi1 ) * DeltaOmega**2 * DeltaD**2
		output -= r/2* scalarProduct( B11 - dSD1phi1, B11 - dSD1phi1 ) * DeltaOmega**2 * DeltaD**2
		
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
		
		RHS0 = np.zeros(  (nO+1,nO,nD,nD) ) 
		RHS1 = np.zeros(  (nO,nO+1,nD,nD) ) 
		
		RHS0 -= BT0 * DeltaOmega**2
		RHS1 -= BT1 * DeltaOmega**2
		
		RHS0 -=  gradAOmega0(  muTilde00   ) * DeltaOmega**2
		RHS0 -=  gradAOmega1(  muTilde01   ) * DeltaOmega**2
		RHS1 -=  gradAOmega0(  muTilde10   ) * DeltaOmega**2
		RHS1 -=  gradAOmega1(  muTilde11   ) * DeltaOmega**2
		
		RHS0 += r*gradAOmega0(  A00   ) * DeltaOmega**2 * DeltaD**2
		RHS0 += r*gradAOmega1(  A01   ) * DeltaOmega**2 * DeltaD**2
		RHS1 += r*gradAOmega0(  A10   ) * DeltaOmega**2 * DeltaD**2
		RHS1 += r*gradAOmega1(  A11   ) * DeltaOmega**2 * DeltaD**2
		
		# Take the splitting adjoint of both E and B 
		ES00, ES01, ES10, ES11 = splittingA(  E00, E01, E10, E11 )
		BS00, BS01, BS10, BS11 = splittingA(  B00, B01, B10, B11 )
		
		RHS0 -= gradD0(  ES00  ) * DeltaOmega**2
		RHS0 -= gradD1(  ES01  ) * DeltaOmega**2
		RHS1 -= gradD0(  ES10  ) * DeltaOmega**2
		RHS1 -= gradD1(  ES11  ) * DeltaOmega**2
		
		RHS0 += r*gradD0(  BS00  ) * DeltaOmega**2 * DeltaD**2
		RHS0 += r*gradD1(  BS01  ) * DeltaOmega**2 * DeltaD**2
		RHS1 += r*gradD0(  BS10  ) * DeltaOmega**2 * DeltaD**2
		RHS1 += r*gradD1(  BS11  ) * DeltaOmega**2 * DeltaD**2
		
		# Solve the system 
		
		solution0, res0 = scspl.cg(LaplaceOp0,RHS0.reshape(((nO+1)*nO*nD*nD)), M = precondOp0 )
		solution1, res1 = scspl.cg(LaplaceOp1,RHS1.reshape((nO*(nO+1)*nD*nD)), M = precondOp1 )
		
		phi0 = solution0.reshape((nO+1,nO,nD,nD)) / ( r * DeltaOmega**2 * DeltaD**2 )
		phi1 = solution1.reshape((nO,nO+1,nD,nD)) / ( r * DeltaOmega**2 * DeltaD**2 )	
		
		endLaplace = time.time()
		print( "Solving the Laplace system: " + str( round( endLaplace - startLaplace, 2) ) + "s." )
		
		if detailStudy:
			objectiveValue[3*counterMain + 1] = objectiveFunctional()
		
		# Projection over a convex set ---------------------------------------------------------
		# It projects on the set Tr(A) + 1/2 |B|^2 <= 0. We reduce to a 1D projection, then use a Newton method with a fixed number of iteration. 
		
		startProj = time.time()
			
		# Computing the derivatives of phi and split them in D 
		dO0phi0, dO1phi0, dO0phi1, dO1phi1 = derivativeOphi()
		dSD0phi0, dSD1phi0, dSD0phi1, dSD1phi1 = derivativeSplittingDphi()
		
		# Compute what needs to be projected
		
		# On A 
		aArray =  dO0phi0 + dO1phi1 + (muTilde00  + muTilde11) / ( r * DeltaD**2 ) 
		
		# On B
		toProjectB00 = dSD0phi0 +  E00 / ( r * DeltaD**2 )
		toProjectB01 = dSD1phi0 +  E01 / ( r * DeltaD**2 )
		toProjectB10 = dSD0phi1 +  E10 / ( r * DeltaD**2 )
		toProjectB11 = dSD1phi1 +  E11 / ( r * DeltaD**2 )
			
		bSquaredArray = np.sum( np.square(toProjectB00) + np.square(toProjectB01) + np.square(toProjectB10) + np.square(toProjectB11) , axis = (-1,-2) ) / 8  
		
		# Compute the array discriminating between the values already on the convex and the others 
		# Value of the objective functional. For the points not in the convex, we want it to vanish.
		projObjective = aArray + bSquaredArray 
		# projDiscriminating is 1 is the point needs to be projected, 0 if it is already in the convex 
		projDiscriminating = np.greater( projObjective, 10**(-16) * np.ones( (nO,nO,nD,nD) ) )
		projDiscriminating = projDiscriminating.astype(int)
		projDiscriminating = projDiscriminating.astype(float)

		# Newton method iteration 
		
		# Value of the Lagrange multiplier. Initialized at 0, not updated if already in the convex set 
		xProj = np.zeros((nO,nO,nD,nD))
		
		for counterProj in range(20) : 
			# Objective functional
			projObjective = aArray + 8 * xProj + np.divide(bSquaredArray,  np.square( 1 - xProj ) )
			# Derivative of the ojective functional 
			dProjObjective = 8 - 2 * np.divide( bSquaredArray, np.power( xProj - 1 , 3) ) 
			# Update of xProj 
			xProj -= np.divide(np.multiply( projDiscriminating, projObjective ), dProjObjective)
				
		# Update of A and B as a result 
		
		A00 = dO0phi0 + muTilde00 / (r * DeltaD**2) + 4*xProj
		A11 = dO1phi1 + muTilde11 / (r * DeltaD**2) + 4*xProj
		
		# Rescale xProj so as it has the same dimension as B and E 
		xProj = np.kron(  xProj.reshape((nO*nO*nD*nD)) , np.ones(4) ).reshape((nO,nO,nD,nD,2,2))	
		
		B00 = np.divide( toProjectB00, (1 - xProj) )
		B01 = np.divide( toProjectB01, (1 - xProj))
		B10 = np.divide( toProjectB10, (1 - xProj))
		B11 = np.divide( toProjectB11, (1 - xProj))	
		
		# Update of the anti-diagonal values of A. It is simply a quadratic maximization.
		
		A01 = dO1phi0 + muTilde01 / ( r * DeltaD**2 ) 
		A10 = dO0phi1 + muTilde10 / ( r * DeltaD**2 )   
			
		# Print the info 
		
		endProj = time.time()
		print( "Pointwise projection: " + str( round( endProj - startProj, 2) ) + "s." )
		
		if detailStudy:
			objectiveValue[3*counterMain + 2] = objectiveFunctional()	
		
		# Gradient descent in (E,muTilde), i.e. in the dual ----------------------------------------- 
			
		# No need to recompute the derivatives of phi  
		
		# Update for muTilde -- no need to update the cross terms, they vanish 
		muTilde00 -= r * DeltaD**2 * ( A00 - dO0phi0 )
		muTilde01 -= r * DeltaD**2 * ( A01 - dO1phi0 )
		muTilde10 -= r * DeltaD**2 * ( A10 - dO0phi1 )
		muTilde11 -= r * DeltaD**2 * ( A11 - dO1phi1 )	

		# Update for E 
		E00 -= r * DeltaD**2 * ( B00 - dSD0phi0 )
		E01 -= r * DeltaD**2 * ( B01 - dSD1phi0 )
		E10 -= r * DeltaD**2 * ( B10 - dSD0phi1 )
		E11 -= r * DeltaD**2 * ( B11 - dSD1phi1 )
		
		# Compute the residuals ------------------------------------------------------------------ 
		
		# For the primal residual, just sum what was the update in the dual 
		primalResidual[counterMain] = DeltaOmega * DeltaD * lin.norm( np.array( [lin.norm( A00 - dO0phi0  ), lin.norm( A01 - dO1phi0  ), lin.norm( A10 - dO0phi1  ), lin.norm( A11 - dO1phi1  ), lin.norm( B00 - dSD0phi0 ), lin.norm( B01 - dSD1phi0 ), lin.norm( B10 - dSD0phi1 ), lin.norm( B11 - dSD1phi1 )])  )
		
		# For the residual, take the RHS of the Laplace system and conserve only BT and the dual variables mu, E 
		
		dualResidualAux0 = np.zeros((nO+1,nO,nD,nD))
		dualResidualAux1 = np.zeros((nO,nO+1,nD,nD))
		
		dualResidualAux0 -= BT0 
		dualResidualAux1 -= BT1 
		
		dualResidualAux0 -=  gradAOmega0(  muTilde00   ) 
		dualResidualAux0 -=  gradAOmega1(  muTilde01   ) 
		dualResidualAux1 -=  gradAOmega0(  muTilde10   ) 
		dualResidualAux1 -=  gradAOmega1(  muTilde11   ) 
		
		# Take the splitting adjoint of both E and B 
		ES00, ES01, ES10, ES11 = splittingA(  E00, E01, E10, E11 )
		
		dualResidualAux0 -= gradD0(  ES00  ) 
		dualResidualAux0 -= gradD1(  ES01  ) 
		dualResidualAux1 -= gradD0(  ES10  ) 
		dualResidualAux1 -= gradD1(  ES11  )
		
		dualResidual[counterMain] = DeltaOmega * r*sqrt( lin.norm( dualResidualAux0 )**2+ lin.norm( dualResidualAux1 )**2    )
		
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
		print(np.max(phi0))
		print(np.min(phi0))
		
		print("Values of A")
		print(np.max(A00))
		print(np.min(A00))
		
		print("Values of mu")
		print(np.max(muTilde00+muTilde11))
		print(np.min(muTilde00+muTilde11))
		
		print("Values of E")
		print(np.max(E00))
		print(np.min(E00))	
		
		print("r")
		print(r)
		
	#****************************************************************************************************
	# End of the program, printing  and returning the results 	#****************************************************************************************************	
	# Mu will be the density which will be plotted in the end. It is equal to both muTilde00 and muTilde11.
	mu = muTilde00[:,:,:,:]
	
	print()
	print( 30*"-" +  " End of the ADMM iterations " + 30*"-"  )
	
	# Measure rho 	
	intMu = np.sum( mu, axis=(-1,-2) )

	print("Difference muTilde")
	print( np.max( np.abs(  muTilde00 - muTilde11  )   ) )
	
	print("Minimal and maximal value of integral of the density")
	print(np.min(intMu))
	print(np.max(intMu))


	print("Maximal and minimal value of the density")
	print(np.min(mu) / DeltaD**2)
	print(np.max(mu) / DeltaD**2)

	print("Final value of the augmentation paramter")
	print(r)

	print("Final value of the objective functional")
	print( objectiveValue[-1] )

	endProgramm = time.time()

	print( "Total time taken by the program: " + str( round( endProgramm - startProgram, 2) ) + "s." )
	
	return mu, E00, E01, E10, E11, objectiveValue, primalResidual, dualResidual
		
	


