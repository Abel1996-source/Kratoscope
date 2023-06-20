### Kratoscope - Real-Data
# Vermare Orson - 30 march 2023

# This programm is used to get rid of sub-surface fluorescence in real-world data.
# It is a hand-made implementation of a relaxed form of the ISRA algorithm.

# Libraries
import numpy as np 
import matplotlib.pyplot as plt
import scipy.signal as sig
import skimage
import time

## Noyau de convolution #######################################################################
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
    eps = 1e-8 # Digital Zero
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

## Algorithme ISRA ###########################################################################
def ISRA(y, H, maxIter, tol, gamma, reg_mode):
    ''' INPUTS : 
        y : 3D image to deconvolve (np.array of size (P1, P2, M))
        H : 3D kernel (np.array of size (2*k+1, 2*k+1, N))
        maxIter : maximum number of iterations (int)
        tol : tolerance for stopping criterion (float)
        gamma : Regularization parameter (float)
        reg_mode : Regularization mode (str) : 'Quad' only

        OUTPUT :
        x : 3D deconvolved image (np.array of size (P1+2*k, P2+2*k, M+N-1))
    '''

    # Initialisation of x (do not use 0)
    P1, P2, M = y.shape
    p1, p2, N = H.shape
    x = np.ones((P1+p1-1, P2+p2-1, M+N-1))
    
    # Creation of error list
    err_list = np.zeros((maxIter+1,))
    err_list[0] = Error(x, y, H, gamma, reg_mode)
    cond_list= np.zeros((maxIter,))

    y_norm = np.linalg.norm(y)
    cond = y_norm + 2*tol

    # Loop
    compt_iter = 0
    while (compt_iter < maxIter) and (cond/y_norm > tol):

        print("------ Step {} ------".format(compt_iter+1))
        step_time = time.time()

        # Positivity Constraint Verification
        if np.any(x<0):
            raise Exception('Non positive element detected in x')

        u, v = U_and_V(x, y, H, reg_mode, gamma)

        # Maximal Step Size
        alpha = init_stepsize(u, v)
        print(" - Initial stepsize = {}".format(alpha))

        # Descent Direction 
        d = descent_direction(x, u, v)

        # Optimal Step
        alpha = armijo(x, y, H, u, v, d, alpha, gamma, beta=0.1, sigma=0.9)
        print(" - Final stepsize = {}".format(alpha))
        # Next estimate
        x = x + alpha*d

        cond = np.linalg.norm(alpha*d) # for stoping criterion
        print("Stop Criterion : ", cond/y_norm)
        cond_list[compt_iter] = cond/y_norm
        
        # Saving intermediate results every 10 iterations
        if not(compt_iter % 10):
            from tifffile import imsave
            imsave('/Users/vermareorson/Desktop/Res-files/'+str(compt_iter)+'.tif', x, photometric='minisblack')
        
        compt_iter += 1
        err_list[compt_iter] = Error(x, y, H, gamma, reg_mode)

        print('>> Error = {:.6f}'.format(err_list[compt_iter]))
        if err_list[compt_iter] > err_list[compt_iter-1]:
            #raise Exception('Warning : Error increased')
            print('---------- WARNING : ERROR INCREASED ----------')
        
        print("--- %.2f seconds ---" % (time.time() - step_time))

    print("------   End   ------")

    plt.figure()
    plt.plot(np.arange(1,maxIter+1), err_list[1:])
    plt.xlabel('Iterations ({})'.format(compt_iter))
    plt.ylabel('Error')
    plt.show()

    plt.figure()
    plt.plot(np.arange(1,maxIter), cond_list[1:])
    plt.xlabel('Iterations ({})'.format(compt_iter))
    plt.ylabel('Stop Criterion')
    plt.show()

    return x

def U_and_V(x, y, H, reg_mode, gamma):
    ''' INPUTS :
        x : 3D deconvolved image (np.array of size (P1+2*k, P2+2*k, M+N-1))
        y : 3D image to deconvolve (np.array of size (P1, P2, M))
        H : 3D kernel (np.array of size (2*k+1, 2*k+1, N))
        reg_mode : Regularization mode (str) : 'Quad' only
        gamma : Regularization parameter (float)

        OUTPUT : 
        u : np.array of size (P1+2*k, P2+2*k, M+N-1)
        v : np.array of size (P1+2*k, P2+2*k, M+N-1)
    '''

    H_t = H[:,:,::-1]

    # Compute U
    Hty = sig.fftconvolve(y, H_t, 'full')

    u = np.where(Hty>0, Hty, 0)

    # Compute V
    Hx = sig.fftconvolve(x, H, 'valid')
    HtHx = sig.fftconvolve(Hx, H_t, 'full')

    if reg_mode == 'Quad':
        v = HtHx - np.where(Hty<0, Hty, 0) + gamma*x
    else:
        raise Exception('reg_mode not valid')
    
    return u, v

def init_stepsize(u, v):
    ''' INPUTS :
        u : np.array of size (P1+2*k, P2+2*k, M+N-1)
        v : np.array of size (P1+2*k, P2+2*k, M+N-1)

        OUTPUT :
        alpha : initial stepsize (float >1)
    '''
    eps = 1e-8 # Digital Zero 
    uv = 1/(1 - (u/(v + eps)))
    grad = v - u 
    try :
        return np.min(uv[(grad>0)])
    except Exception:
        print("init_stepsize : No satisfying value encountered")
        return 1

def descent_direction(x, u, v):
    ''' INPUTS :
        x : 3D deconvolved image (np.array of size (P1+2*k, P2+2*k, M+N-1))
        u : np.array of size (P1+2*k, P2+2*k, M+N-1)
        v : np.array of size (P1+2*k, P2+2*k, M+N-1)

        OUTPUT :
        d : descent direction (np.array of size (P1+2*k, P2+2*k, M+N-1))
    '''
    eps = 1e-8 # Digital Zero 
    return (x/(v + eps))*(u - v)

def Error(res, sample, H, gamma=0, reg_mode='Quad'):
    ''' INPUTS :
        res : Estimated convoluted image (np.array of size (P1, P2, M)
        sample : Reference input image (np.array of size (P1, P2, M)
        H : 3D kernel (np.array of size (2*k+1, 2*k+1, N))
        gamma : Regularization parameter (float)
        reg_mode : Regularization mode (str) : 'Quad' only

        OUTPUT :
        the error criteria to minimize (float)
    '''
    
    if reg_mode == 'Quad':
        elem = np.size(res) # number of elements
        sim = sig.fftconvolve(res, H, 'valid')
        j1 = (np.sum((sample-sim)**2))/(2*elem) # invariant to the number of elements
        j2 = np.sum((res)**2)/elem
        print("J1 : {} -- J2 : {} -- J1/J2 : {}".format(j1, j2, j1/j2))
        return (j1 + gamma*j2)
    else :
        raise Exception('reg_mode not valid')

def armijo(x, y, H, u, v, d, alpha_0, gamma, beta, sigma):
    ''' INPUTS :
        x : 3D deconvolved image (np.array of size (P1+2*k, P2+2*k, M+N-1))
        y : 3D image to deconvolve (np.array of size (P1, P2, M))
        H : 3D kernel (np.array of size (2*k+1, 2*k+1, N))
        u : np.array of size (P1+2*k, P2+2*k, M+N-1)
        v : np.array of size (P1+2*k, P2+2*k, M+N-1)
        d : descent direction (np.array of size (P1+2*k, P2+2*k, M+N-1))
        alpha_0 : initial stepsize (float >1)
        gamma : Regularization parameter (float)
        beta : reduction factor (float usually between 0.1 and 0.5)
        sigma : descent factor (float usually between 1e-5 and 1e-1)

        OUTPUT :
        alpha : ideal stepsize (float <alpha_0)
    '''
    J = Error(x, y, H, gamma, reg_mode)
    x_new = x + alpha_0*d
    J_new = Error(x_new, y, H, gamma, reg_mode)

    grad_d = np.sum((v-u)*d)/np.size(d)
    m = 0
    while (not(J - J_new >= -sigma*(beta**m)*alpha_0*grad_d)):
        print("    {} Intermediate Stepsize {}".format(m, alpha_0*(beta**m)))
        print(J-J_new, -sigma*(beta**m)*alpha_0*grad_d, (J-J_new)+(sigma*(beta**m)*alpha_0*grad_d))
        m += 1
        x_new = x + (beta**m)*alpha_0*d
        J_new = Error(x_new, y, H, gamma, reg_mode)
    return alpha_0*(beta**m)

# Real Data ###################################################################################

def load_data():
    file = "/Users/vermareorson/dataProjet/stackSubtracted543-translationcorrected.tif"

    images = skimage.io.imread(file)[::2, ::3, ::3]

    print('Before scaling', images.max(), images.min(), images.dtype)
    images = images/images.max()
    print('After scaling', images.max(), images.min(), images.dtype)

    Q,N,P = images.shape
    images_reverse = np.zeros((N,P,Q))
    for i in range(Q):
        images_reverse[:,:,i] = images[i,:,:]

    return images_reverse

## Main Script ################################################################################
start_time = time.time()

image = load_data()
print("Image shape : {}".format(image.shape))

# Parameter Choice

d_A = 0.98885054
d_sigma = 2.11630693
P1, P2, M = image.shape
N = image.shape[2]
maxIter = 2
tol = 1e-4
gamma = 1e-9
reg_mode = 'Quad'

# Creation of matrices
ker = Kernel(N, d_A, d_sigma, type='both')
print("Kernel shape : {}".format(ker.shape))
print("Sum Kernel ", np.sum(ker))
k = int((ker.shape[0]-1)/2)

# Solving (ISRA)
res = ISRA(image, ker[:,:,::-1], maxIter, tol, gamma, reg_mode)

# Computing the Residual
res_sim = sig.fftconvolve(res, ker[:,:,::-1], mode='valid')
print("Results Simul shape : {}".format(res_sim.shape))
left_over = image - res_sim #for choice of gamma

res = res[k:P1+k, k:P2+k, :M] # Troncation !
print("Results shape : {}".format(res.shape))
print('Before scaling', res.max(), res.min(), res.dtype)
res = res/res.max()
print('After scaling', res.max(), res.min(), res.dtype)

# Saving the final result
from tifffile import imsave
imsave('/Users/vermareorson/Desktop/Res-files/FINAL.tif', res, photometric='minisblack')

print('d_A = {} -- d_sigma = {} -- gamma = {}'.format(d_A, d_sigma, gamma))
print('maxIter = {} -- tol = {}'.format(maxIter, tol))

print(" - Kernel global min/max : {:.2f} / {:.2f}".format(ker.min(), ker.max()))
print(" - Image global min/max : {:.2f} / {:.2f}".format(image.min(), image.max()))
print(" - Results global min/max : {:.2f} / {:.2f}".format(res.min(), res.max()))

print(">> All positive Results : {}".format(np.all(res>0)))

print("--- %.2f seconds ---" % (time.time() - start_time))

# Display
if (0):
    for n in range(0, ker.shape[2], 4):
        plt.figure(figsize=(5, 6))
        plt.imshow(ker[:,:,n], cmap='gray')
        plt.title('Z = {}'.format(ker.shape[2] - n))
        plt.colorbar()
        plt.show()
if (0):
    for m in range(0, res.shape[2], 5):

        plt.figure(figsize=(12, 4))

        plt.subplot(1,3,1)
        plt.imshow(image[:,:,m], cmap='gray')
        plt.title('Z = {}'.format(image.shape[2] - m))
        plt.colorbar()

        plt.subplot(1,3,2)
        plt.imshow(res[:,:,m], cmap='gray')
        plt.title('Z = {}'.format(res.shape[2] - m))
        plt.colorbar()

        plt.subplot(1,3,3)
        plt.imshow(left_over[:,:,m], cmap='gray')
        plt.title('Z = {}'.format(left_over.shape[2] - m))
        plt.colorbar()
        plt.show()

if (0): 
    #Profils Y=150 et Z=332
    prof_res = res[150,:,60]
    prof_image = image[150,:,60]
    prof_leftover = left_over[150,:,60]

    plt.figure()
    plt.plot(prof_image, label='Image acquise')
    plt.plot(prof_res, label='Résultats')
    plt.plot(prof_leftover, label='Résidus')
    plt.xlabel('X')
    plt.ylabel('Intensité pixels')
    plt.legend()
    plt.show()