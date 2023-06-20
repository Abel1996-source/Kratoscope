# Libraries
import numpy as np 
import matplotlib.pyplot as plt
import scipy.signal as sig
import skimage
import time
from scipy.optimize import minimize
from numpy.random import default_rng

## Sample #######################################################################
def create_sample_matrix(P1, P2, M, k, alpha):
    '''  INPUTS :
        P1 : width of sample (int >0)
        P2 : length of sample (int >0)
        M : depth of sample (int >0)
        k : half-size of kernel (int >0)
        alpha : angle of the plane (float)

        OUTPUT :
        sample : np.array of size (P1+2*k, P2+2*k, M)
    '''
    sample = np.zeros((P1+2*k, P2+2*k, M))
    for m in range(M):
        if int(240+k-np.tan(alpha)*m) >=0 :
            sample[65+k:400+k, int(240+k-np.tan(alpha)*m):int(260+k-np.tan(alpha)*m), m] = 1
    return sample

# Real Data #####################################################################
def load_data():

    file = "/Users/vermareorson/dataProjet/Subtracted_fantome_parafblanche/09-55-23_014-fantome_Z1_p2--Subtracted.tif"
    images = skimage.io.imread(file)[350:850, 650:950]

    return images

## Noyau de convolution #########################################################
def Kernel(N, d_A, d_sigma, type):
    ''' INPUTS : 
        N : depth of kernel (int >0)
        d_A : intensity attenuation of one slice (int <1)
        d_sigma : standard deviation of the gaussian scatter (int >0)
        type : 'scatter' 'attenuation' 'both' (str)

        OUTPUT :
        ker : np.array of size (2k+1, 2*k=1, N)

        Local variables :
        k : half-size of kernel (int >0) such that the scatter doesn't overflow
        thresh : overflow limit, enables to control the size (x,y) of the kernel
    '''
    eps = 1e-6 # Digital Zero
    thresh = 1e-5 # overflow limit

    try :
        k = int(np.sqrt(2*(1-N)*d_sigma**2*np.log(eps + (thresh*2*np.pi*(N-1)*d_sigma**2)/(d_A**(N-1)))))
    except Exception:
        print("Overflow limit to high : decrease thresh")
        return None
    
    P = 2*k+1
    ker = np.zeros((P,P,N))

    ker[k,k,0] = 1 # First Slice is a dirac

    for n in range(1,N):
        if type=='scatter':
            # Integral normalized
            ker[:,:,n] = np.array([np.exp(-((i-k)**2 + (j-k)**2)/(2*(n+eps)*(d_sigma**2)))/(2*np.pi*(n+eps)*d_sigma**2) for i in range (P) for j in range(P)]).reshape((P,P))
            # or not 
            #ker[:,:,n] = np.array([np.exp(-((i-k)**2 + (j-k)**2)/(2*(n+eps)*(d_sigma**2))) for i in range (P) for j in range(P)]).reshape((P,P))
        elif type=='attenuation':
            ker[:,:,n] = np.ones((P,P))*(d_A**n) 
        elif type=='both':
            ker[:,:,n] = np.array([(d_A**n)*np.exp(-((i-k)**2 + (j-k)**2)/(2*(n+eps)*(d_sigma**2)))/(2*np.pi*(n+eps)*d_sigma**2) for i in range (P) for j in range(P)]).reshape((P,P))
        else :
            raise Exception('Type not valid')
    return ker

#crit ###############################################################################################################
def crit(x):
    ''' INPUTS : 
        x : simulated image (np.array of size (P1+2*k, P2+2*k, 1))
        image (global variable) : reference image (np.array of size (P1, P2, 1))

        OUTPUT :
        val : squared sum of differences normalized by number of elements in the simulated image
    '''

    global image # Reference image
    alpha, d_A, d_sigma = x

    P1, P2, M = image.shape[0], image.shape[1], int(240/np.tan(alpha))
    print('M = {}'.format(M))
    N = M
    ker = Kernel(N, d_A, d_sigma, type='both')
    print("Kernel Size {}".format(ker.shape))

    k = int((ker.shape[0]-1)/2)
    
    #here we are create matrix sample
    sample = create_sample_matrix(P1, P2, M, k, alpha)
    print("Sample Size {}".format(sample.shape))

    sim = sig.convolve(sample, ker[:,:,::-1], mode='valid')
    sim = sim/sim.max()
    print("Simulation Size {}".format(sim.shape))
    
    val = np.sum((image - sim[:,:,0])**2)/np.prod(sim.shape) #Normalized for invariance to sim shape
    
    print("alpha = {}, d_A= {:.3f}, d_sigma= {:.3f}".format(alpha, d_A, d_sigma))
    print('--- Critère : {}'.format(val))
    return val