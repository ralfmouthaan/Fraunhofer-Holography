# Ralf Mouthaan
# University of Cambridge
# March 2022
#
# Gerchberg-Saxton
# TODO:
#   * Bandwidth limiting
#   * Power smoothing
#   * Fienup
#   * FIDOC
#   * Different starting phase profiles
#   * Apply random starting phase to the hologram, or the replay field?

import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import DRY_Holography as DRY

def main():

    # User-defined parameters
    NoIters = 100
    Quantisation = 256
    ErrMetric = 'PSNR'
    bolBandLimit = 1
    bolPowerSmooth = 0
    Target = DRY.LoadImage('Test Images/Mandrill.tiff', Quantisation)

    # Set up
    Height, Width = Target.shape
    arrErr = np.zeros(NoIters)
    arrTime = np.zeros(NoIters)
    Replay = np.exp(1j*2*np.pi*np.random.rand(Height, Width)) # Randomised initial replay field

    # Target is normalised based on the assumption of a hologram of only ones.
    # But, if the hologram is bandwidth-limited, then the outer half is zeros.
    # So, renormalise to take this into account.
    if bolBandLimit == 1:
        Target = Target/4
        Replay = Replay/4

    # GS algorithm
    t0 = timer()
    for ii in range(0, NoIters):
        
        Replay = Target*np.exp(1j*np.angle(Replay))
        Holo = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(Replay)))*np.sqrt(Width*Height)
        Holo = QuantisePhase(Holo, bolBandLimit, bolPowerSmooth, Quantisation) # Note, this function is different to GS.
        Replay = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(Holo)))/np.sqrt(Width*Height)

        arrErr[ii] = DRY.CalculateErrorMetric(Target, Replay, ErrMetric)
        arrTime[ii] = timer() - t0

    Replay = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(Holo)))/np.sqrt(Width*Height)
    
    # Plot Target, Holo, Replay
    DRY.SaveErrToFile("Gerchberg-Saxton - " + ErrMetric + ".txt", arrTime, arrErr, ErrMetric)
    DRY.PlotTargetHoloReplay(Target, Holo, Replay)
    DRY.PlotErrVsTime(arrTime, arrErr, ErrMetric)
    plt.show()

def QuantisePhase(Holo, bolBandLimit, bolPowerSmooth, Quantisation):

    if Quantisation == 'Continuous':
        Quantisation = 0
    elif Quantisation == 'Multi-Level':
        Quantisation = 256
    elif Quantisation == 'Binary':
        Quantisation = 2

    # Power-spectrum smoothing
    if bolPowerSmooth == 1:
        Holo[Holo > 1] = np.exp(1j*np.angle(Holo[Holo > 1]))
    else:
        Holo = np.exp(1j*np.angle(Holo))

    # Bandwidth-limiting
    if bolBandLimit == 1:
        Height, Width = Holo.shape
        Mask = np.ones((Height, Width))
        Mask[int(Height/4):int(Height*3/4)-1,int(Width/4):int(Width*3/4)-1] = 0
        Holo[Mask > 0] = 0
    
    # Quantising
    if Quantisation > 0:
        angles = (np.angle(Holo) + np.pi)/2/np.pi * Quantisation
        angles = np.round(angles)/Quantisation*2*np.pi
        Holo = np.abs(Holo)*np.exp(1j*angles)

    return Holo

if __name__ == "__main__":
    main()