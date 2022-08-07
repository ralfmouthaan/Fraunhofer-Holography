# Ralf Mouthaan
# University of Cambridge
# April 2022
#
# Don't repeat yourself (DRY) holography functions.
# A library of helper functions

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr

def LoadImage(strFile, Quantisation):

    # Load Image,
    # Convert to grayscale,
    # Convert to numpy array
    Img = Image.open(strFile)
    Img = ImageOps.grayscale(Img)
    Img = np.asarray(Img)
    
    # Implement symmetries if a binary device is considered
    if Quantisation == "Binary" or Quantisation == 2:
        Img = ImposeConjugateSymmetry(Img)

    # Normalise for subsequent power conservation stuff
    Img = NormaliseSumPower(Img)

    return Img

def NormaliseSumPower(Holo):

    Height, Width = Holo.shape

     # For big fields we seem to sometimes get rounding errors, so do it until it's good.
    while np.abs(CalculateSumPower(Holo) - 1) > 1e-10:
        Holo = Holo / np.sqrt(CalculateSumPower(Holo))
    Holo = Holo * np.sqrt(Height*Width)

    return Holo

def ImposeConjugateSymmetry(Img):

    Height, Width = Img.shape
    Img[int(Height/2):Height - 1,:] = np.flipud(np.fliplr(Img[0:int(Height/2)-1,:]))
    return Img

def QuantisePhase(Holo, Quantisation):

    # Holo is the hologram matrix
    # Quantisation is the type of quantisation required:
    #   * "Continuous": Continuous quantisation
    #   * "Multi-Level": 256-level quantisation
    #   * "Binary": 2-level quantisation
    #   * 0: Continuous quantisation - amplitudes set to unity
    #   * 2: Binary quantisation
    #   * 256: 256-level quantisation
    #   * and so forth

    if Quantisation == 'Continuous':
        Quantisation = 0
    elif Quantisation == 'Multi-Level':
        Quantisation = 256
    elif Quantisation == 'Binary':
        Quantisation = 2

    if Quantisation == 0:
        Holo = np.exp(1j*np.angle(Holo))
    else:
        angles = (np.angle(Holo) + np.pi)/2/np.pi * Quantisation
        angles = np.round(angles)/Quantisation*2*np.pi
        Holo = np.exp(1j*angles)

    return Holo

def CalculateSumPower(X):

    # Calculates total power in field

    return np.sum(np.sum(np.abs(X)**2))

def CalculateMSE(X,Y):

    # Calculates the phase-insensitive MSE
    # Tested to be the same as skimage's MSE

    if X.shape != Y.shape:
        raise Exception("CalculateMSE: Input matrices must have same shape")

    Height, Width = X.shape
    RetVal = np.sum(np.sum((np.abs(X) - np.abs(Y))**2)) / Height / Width
    return RetVal

def CalculatePSNR(X,Y):

    # Calculates PSNR
    # Tested to be the same as skimage's psnr.

    maxX = np.max(np.max(np.abs(X)))
    maxY = np.max(np.max(np.abs(Y)))
    maxmax = max(maxX, maxY)

    if X.shape != Y.shape:
        raise Exception("CalculatePSNR: Input matrices must have same shape")

    return 20*np.log10(maxmax) - 10*np.log10(CalculateMSE(X,Y))

def CalculateSSIM(X,Y):

    # Note, this assumes E-field are passed in, and then squares these to get intensities.
    # These seems like the sensible thing to do.

    if X.shape != Y.shape:
        raise Exception("CalculateSSIM: Input matrices must have same shape")

    return ssim(np.abs(X)**2,np.abs(Y)**2)

def CalculateErrorMetric(X, Y, Metric):

    if Metric == 'MSE':
        return CalculateMSE(X,Y)
    elif Metric == 'PSNR':
        return CalculatePSNR(X,Y)
    elif Metric == 'SSIM':
        return CalculateSSIM(X,Y)

def PlotTargetHoloReplay(Target, Holo, Replay):

    fig, axs = plt.subplots(1,3)
    axs[0].imshow(Target, cmap='gray')
    axs[0].axis('off')
    axs[0].set_title('Target')

    axs[1].imshow(np.angle(Holo), cmap='gray')
    axs[1].axis('off')
    axs[1].set_title('Holo')

    axs[2].imshow(np.abs(Replay), cmap='gray')
    axs[2].axis('off')
    axs[2].set_title('Replay')

def PlotErrVsIterNo(arrErr, strErrMetric):

    plt.figure()
    plt.plot(range(1,len(arrErr)+1), arrErr)
    plt.xlabel('Iteration No.')
    if strErrMetric == 'MSE':
        plt.ylabel('MSE')
        plt.ylim(0, max(arrErr)*1.1)
    elif strErrMetric == 'PSNR':
        plt.ylabel('PSNR')
    elif strErrMetric == 'SSIM':
        plt.ylabel('SSIM')

def PlotErrVsTime(arrTime, arrErr, strErrMetric):

    plt.figure()
    plt.plot(arrTime, arrErr)
    plt.xlabel('Time (s)')
    if strErrMetric == 'MSE':
        plt.ylabel('MSE')
        plt.yscale('log')
    elif strErrMetric == 'PSNR':
        plt.ylabel('PSNR')
    elif strErrMetric == 'SSIM':
        plt.ylabel('SSIM')

def SaveErrToFile(Filename, arrTime, arrErr, strErrMetric):

    f = open(Filename, 'w')

    f.write('Time(s) \t ' + strErrMetric + '\n')
    for ii in range(0, len(arrErr)):
        f.write(f'{arrTime[ii]:0.3f} \t {arrErr[ii]:0.3f}\n')
    f.close