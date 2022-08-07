# Ralf Mouthaan
# University of Cambridge
# March 2022
#
# Classic Gerchberg-Saxton.
# Based on the original paper I suppose, but implemented for CGH.
# Allows different types of quantisation to be applied to cater to different SLMs

import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import DRY_Holography as DRY

def main():

    # User-defined parameters
    NoIters = 100
    Quantisation = 256
    ErrMetric = 'PSNR'
    Target = DRY.LoadImage('Test Images/Mandrill.tiff', Quantisation)

    # Set up
    Height, Width = Target.shape
    arrErr = np.zeros(NoIters)
    arrTime = np.zeros(NoIters)
    Replay = np.exp(1j*2*np.pi*np.random.rand(Height, Width)) # Randomised initial replay field

    # GS algorithm
    t0 = timer()
    for ii in range(0, NoIters):
        
        Replay = Target*np.exp(1j*np.angle(Replay))
        Holo = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(Replay)))*np.sqrt(Width*Height)
        Holo = DRY.QuantisePhase(Holo, Quantisation)
        Replay = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(Holo)))/np.sqrt(Width*Height)

        arrErr[ii] = DRY.CalculateErrorMetric(Target, Replay, ErrMetric)
        arrTime[ii] = timer() - t0

    Replay = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(Holo)))/np.sqrt(Width*Height)

    if np.abs(DRY.CalculateSumPower(Replay) - Height*Width) > 1e-3:
        raise Exception("GS: Replay power is incorrect")
    if np.abs(DRY.CalculateSumPower(Holo) - Height*Width) > 1e-3:
        raise Exception("GS: Holo power is incorrect")
    if np.abs(DRY.CalculateSumPower(Target) - Height*Width) > 1e-3:
        raise Exception("GS: Target power is incorrect")

    # Plot Target, Holo, Replay
    DRY.SaveErrToFile("Gerchberg-Saxton - " + ErrMetric + ".txt", arrTime, arrErr, ErrMetric)
    DRY.PlotTargetHoloReplay(Target, Holo, Replay)
    DRY.PlotErrVsTime(arrTime, arrErr, ErrMetric)
    plt.show()

if __name__ == "__main__":
    main()