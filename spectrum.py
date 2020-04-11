import numpy as np
import pandas as pd

# import optimization


gm = []
DFTgm = []
Ngm = []
DTgm = np.genfromtxt('/Volumes/GoogleDrive/My Drive/Doktor/CCE Birla Carbon/CCE Milling Tower/Earthquakes/DT.txt')
Tgm = []
dfgm = []
dwgm = []
fNiquist = []
for EQ in range(1, 141):
    gm.append(np.genfromtxt(
        '/Volumes/GoogleDrive/My Drive/Doktor/CCE Birla Carbon/CCE Milling Tower/Earthquakes/' + str(EQ) + '.txt',
        dtype=float))
    DFTgm.append(np.fft.fft(gm[EQ - 1]))
    Ngm.append(len(gm[EQ - 1]))
    Tgm.append(Ngm[EQ - 1] * DTgm[EQ - 1])
    dfgm.append(1 / (Tgm[EQ - 1]))
    dwgm.append(dfgm[EQ - 1] * 2 * np.pi)
    fNiquist.append(Ngm[EQ - 1] / 2 * dfgm[EQ - 1])

data = {'Time Step [s]': DTgm,
        'Number Steps': Ngm,
        'Length record [s]': Tgm,
        'Frequency step DFT [Hz]': dfgm,
        'RadFreq step DFT [rad/s]': dwgm,
        'Niquist Freq DFT [Hz]': fNiquist,
        'Earthquake': gm,
        'Fourier Spectrum': DFTgm}

EQdf = pd.DataFrame(data)

wmin = 0.1
wmax = 150
dw = 0.05
wrange = np.arange(wmin, wmax, dw)

PSDlin = []
for eq in range(len(Ngm)):
    wfreqRange = np.arange(0,
                           EQdf.loc[eq, 'RadFreq step DFT [rad/s]'] * EQdf.loc[eq, 'Number Steps'],
                           EQdf.loc[eq, 'RadFreq step DFT [rad/s]'])
    PSDrange = np.abs(EQdf.loc[eq, 'Fourier Spectrum']) ** 2
    PSDlin.append(np.interp(wrange, wfreqRange[0:len(PSDrange)], PSDrange))

eq = 100
# plt.plot(wrange,PSDlin[eq])
AvgPSD = np.sum(PSDlin, axis=0) / len(Ngm)

np.savetxt('spectrum.txt', AvgPSD)
np.savetxt('FreqRange.txt', wrange)

# spectrum = optimization.excitation(AvgPSD,wrange)

# print(spectrum.spectrum)
# print(EQdf)
