### Krato - Calibration
# Vermare Orson - 30 march 2023

# This program is used to determine the best parameters (d_A, d_sigma) of the PSF as well as (alpha)
# from a reference image (tilted plastic plane in paraffin). 
# It is based on the Nelder-Mead Simplex algorithm.

# Libraries
import numpy as np 
import matplotlib.pyplot as plt
import scipy.signal as sig
import time
from scipy.optimize import minimize
from Krato import create_sample_matrix
from Krato import load_data
from Krato import Kernel
from Krato import crit


# Main Script ##########################################################################
start_time = time.time()

image = load_data()
image = (image - image.min())/(image.max() - image.min())
print("Input Image Size {}".format(image.shape))
print("--- Image import : %.2f seconds ---" % (time.time() - start_time))

# Initialization
d_A0 = 0.9888505365097319 
d_sigma0 = 2.116306928082876
alpha0 = 1
x0 = [alpha0, d_A0, d_sigma0]

# Boundaries
bnds = ((1e-1, np.pi/2-1e-1), (0, 1), (0, None))

# Identification with Nelder-Mead
res = minimize(crit, x0, method='Nelder-Mead', bounds=bnds)
print(res)

print("--- Nelder-Mead : %.2f seconds ---" % (time.time() - start_time))

# Display of the results
alpha, d_A, d_sigma = res.x
alpha, d_A, d_sigma = x0
print('d_A = {} -- d_sigma = {}'.format(d_A, d_sigma))

P1, P2, M = image.shape[0], image.shape[1], int(240/np.tan(alpha)) 
N = M
ker = Kernel(N, d_A, d_sigma, type='both')
k = int((ker.shape[0]-1)/2)
sample = create_sample_matrix(P1, P2, M, k, alpha)
sim = sig.convolve(sample, ker[:,:,::-1], mode='valid')
sim = sim/sim.max()

print("Kernel Size {}".format(ker.shape))
print("Sample Size {}".format(sample.shape))
print("Simulation Size {}".format(sim.shape))

if (0):
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(image, cmap='gray')
    plt.title('Reference Image')
    plt.colorbar()
    plt.subplot(1,2,2)
    plt.imshow(sim, cmap='gray')
    plt.title('Simulation Result')
    plt.colorbar()
    plt.show()

if (1):
    #Profils Y=250 et Z=332
    prof_res = sim[250,:,0]
    prof_image = image[250,:]

    plt.figure()
    plt.plot(prof_image, label='Image de référence')
    plt.plot(prof_res, label='Résultats de la recherche')
    plt.xlabel('X')
    plt.ylabel('Intensité pixels')
    plt.legend()
    plt.show()