# Ralf Mouthaan
# University of Cambridge
# March 2022
#
# Gerchberg-Saxton
# TODO:
#   * Should my error metrics look at E-field, or intensity?
#   * Impose target symmetry for binary holograms
#   * SSIM
#   * Bandwidth limiting
#   * Smooth phase profile
#   * Stick lots into functions to tidy up
#   * Make sure y axis always goes to zero.

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

def main():

    NoIters = 100
    ErrMetric = 'MSE'
    Quantisation = 'Binary'

    arrErr = np.zeros(NoIters)

    # Load Target
    Target = Image.open("Test Images/SailingBoat.tiff")
    Target = ImageOps.grayscale(Target)
    Target = np.asarray(Target)
    Height, Width = Target.shape
    while np.abs(CalculateSumPower(Target) - 1) > 1e-10: # For big fields we seem to sometimes get rounding errors
        Target = Target / np.sqrt(CalculateSumPower(Target))
    Target = Target * np.sqrt(Height*Width)

    # Initial replay field
    Replay = np.exp(1j*2*np.pi*np.random.rand(Height, Width))

    # GS algorithm
    for ii in range(0, NoIters):
        
        Replay = Target*np.exp(1j*np.angle(Replay))
        Holo = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(Replay)))*np.sqrt(Width*Height)
        if Quantisation == 'Continuous':
            Holo = np.exp(1j*np.angle(Holo))
        elif Quantisation == 'Multi-Level':
            angles = np.angle(Holo)/2/np.pi * 255
            angles = np.round(angles)/255*2*np.pi
            Holo = np.exp(1j*angles)
        elif Quantisation == 'Binary':
            angles = np.angle(Holo)/np.pi
            angles = np.round(angles)*np.pi
            Holo = np.exp(1j*angles)
        Replay = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(Holo)))/np.sqrt(Width*Height)
        if ErrMetric == 'MSE':
            arrErr[ii] = CalculateMSE(Target, Replay)
        elif ErrMetric == 'PSNR':
            arrErr[ii] = CalculatePSNR(Target, Replay)

    Replay = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(Holo)))/np.sqrt(Width*Height)
    
    # Plotting
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

    plt.figure()
    plt.plot(range(1,NoIters+1), arrErr)
    plt.xlabel('No. Iterations')
    if ErrMetric == 'MSE':
        plt.ylabel('MSE')
    elif ErrMetric == 'PSNR':
        plt.ylabel('PSNR')

    plt.show()

def CalculateSumPower(X):

    # Calculates total power in field

    return np.sum(np.sum(np.abs(X)**2))

def CalculateMSE(X,Y):

    # Calculates the phase-insensitive MSE

    if X.shape != Y.shape:
        raise Exception("CalculateMSE: Input matrices must have same shape")

    Height, Width = X.shape
    RetVal = np.sum(np.sum((np.abs(X) - np.abs(Y))**2)) / Height / Width
    return RetVal

def CalculatePSNR(X,Y):

    # Calculates PSNR

    maxX = np.max(np.max(np.abs(X)))
    maxY = np.max(np.max(np.abs(Y)))
    maxmax = max(maxX, maxY)

    if X.shape != Y.shape:
        raise Exception("CalculateMSE: Input matrices must have same shape")

    return 20*np.log10(maxmax) - 20*np.log10(CalculateMSE(X,Y))

if __name__ == "__main__":
    main()