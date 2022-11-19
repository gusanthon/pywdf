#  As described in : 
#  https://dafx2020.mdw.ac.at/proceedings/papers/DAFx20in22_paper_39.pdf

import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import scipy
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from circuit import Circuit
from wdf import *


def get_closest(d, search_key):
    if d.get(search_key):
        return search_key, d[search_key]
    key = min(d.keys(), key=lambda key: abs(key - search_key))
    return key, d[key]

def plot_magnitude_response(circuit, label="magnitude", c="tab:blue", title=""):
    x = circuit.get_impulse_response()
    w, h = scipy.signal.freqz(x, 1, 4096)
    H = 20 * np.log10(np.abs(h))
    f = w / (2 * np.pi) * circuit.fs
    ax = plt.subplot(111)
    ax.plot(f, H)
    ax.semilogx(f, H, label=label, color=c)
    plt.ylabel("Amplitude [dB]")
    plt.xlabel("Frequency [hz]")
    plt.title(title + "Magnitude response")
    plt.grid()
    plt.legend()


class RCA_MK2_SEF(Circuit):
    
    def __init__(self, sample_rate: int, highpass_cutoff: float, lowpass_cutoff: float) -> None:

        self.HP_vals = {
            0 : {'C' : 9999999, 'L' : 9999999},
            175 : {'C' : 1.6e-6, 'L' : 255.6e-3},
            248 : {'C' : 1.15e-6, 'L' : 176.9e-3},
            352 : {'C' : .8e-6, 'L' : 126.4e-3},
            497 : {'C' : .57e-6, 'L' : 176.9e-3},
            699 : {'C' : 1.15e-6, 'L' : 90.11e-3},
            1002 : {'C' : .272e-6, 'L' : 44.56e-3},
            1411 : {'C' : .2e-6, 'L' : 31.79e-3 },
            2024 : {'C' : .15e-6, 'L' : 21.77e-3},
            2847 : {'C' : .1e-6, 'L' : 15.63e-3 },
            3994 : {'C' : .071e-6, 'L' : 11.18e-3}
        }

        self.LP_vals = {
            175 : {'C' : 3.22e-6, 'L' : 511.1e-3 },
            245 : {'C' : 2.3e-6, 'L' : 365.2e-3 },
            350 : {'C' : 1.6e-6, 'L' : 255.6e-3 },
            499 : {'C' : 1.15e-6, 'L' : 178.6e-3 },
            703 : {'C' : .8e-6, 'L' : 126.4e-3 },
            996 : {'C' : .57e-6, 'L' : 90.02e-3 },
            1408 : {'C' : .4e-6, 'L' : 63.54e-3},
            1989 : {'C' : .272e-6, 'L' : 45.08e-3},
            2803 : {'C' : .2e-6, 'L' : 32.13e-3 },
            3992 : {'C' : .15e-6, 'L' : 22.38e-3 },
            999999 : {'C' : 1e-15, 'L' : 1e-15}
        }

        self.fs = sample_rate

        self.Z_input = 560
        self.Z_output = 560

        self.highpass_cutoff, HP_vals = get_closest(self.HP_vals, highpass_cutoff)
        self.lowpass_cutoff, LP_vals = get_closest(self.LP_vals, lowpass_cutoff)

        self.C_HP = HP_vals['C']
        self.L_HP = HP_vals['L']

        self.C_LP = LP_vals['C']
        self.L_LP = LP_vals['L']

        self.Rt = Resistor(self.Z_output)
        self.L_LPm2 = Inductor(self.L_LP, self.fs)

        self.S8 = SeriesAdaptor(self.L_LPm2, self.Rt)
        self.C_LPm1 =Capacitor(self.C_LP, self.fs)

        self.P4 = ParallelAdaptor(self.C_LPm1, self.S8)
        self.L_LPm1 = Inductor(self.L_LP, self.fs)

        self.S7 = SeriesAdaptor(self.L_LPm1, self.P4)
        self.L_LP2 = Inductor(self.L_LP, self.fs)

        self.S6 = SeriesAdaptor(self.L_LP2, self.S7)
        self.C_LP1 = Capacitor(self.C_LP, self.fs)

        self.P3 = ParallelAdaptor(self.C_LP1, self.S6)
        self.L_LP1 = Inductor(self.L_LP, self.fs)

        self.S5 = SeriesAdaptor(self.L_LP1, self.P3)
        self.C_HP2 = Capacitor(self.C_HP, self.fs)

        self.S4 = SeriesAdaptor(self.C_HP2, self.S5)
        self.L_HP1 = Inductor(self.L_HP, self.fs)

        self.P2 = ParallelAdaptor(self.L_HP1, self.S4)
        self.C_HP1 = Capacitor(self.C_HP, self.fs)

        self.S3 = SeriesAdaptor(self.C_HP1, self.P2)
        self.C_HPm2 = Capacitor(self.C_HP, self.fs)
        
        self.S2 = SeriesAdaptor(self.C_HPm2, self.S3)
        self.L_HPm = Inductor(self.L_LP, self.fs)

        self.P1 = ParallelAdaptor(self.L_HPm, self.S2)
        self.C_HPm1 = Capacitor(self.C_HP, self.fs)

        self.S1 = SeriesAdaptor(self.C_HPm1, self.P1)
        self.Rin = Resistor(self.Z_input)

        self.S0 = SeriesAdaptor(self.Rin, self.S1)
        self.Vin = IdealVoltageSource(self.S0)

        super().__init__(self.Vin, self.Vin, self.Rt)

    def _set_HP_components(self, C, L):
        self.C_HP1.set_capacitance(C)
        self.C_HP2.set_capacitance(C)
        self.C_HPm1.set_capacitance(C)
        self.C_HPm2.set_capacitance(C)

        self.L_HP1.set_inductance(L)
        self.L_HPm.set_inductance(L)

    def _set_LP_components(self, C, L):
        self.C_LP1.set_capacitance(C)
        self.C_LPm1.set_capacitance(C)

        self.L_LP1.set_inductance(L)
        self.L_LP2.set_inductance(L)
        self.L_LPm1.set_inductance(L)
        self.L_LPm2.set_inductance(L)

    def set_highpass_cutoff(self, new_cutoff):
        self.highpass_cutoff, vals = get_closest(self.HP_vals, new_cutoff)
        self._set_HP_components(vals['C'], vals['L'])

    def set_lowpass_cutoff(self, new_cutoff):
        self.lowpass_cutoff, vals = get_closest(self.LP_vals, new_cutoff)
        self._set_LP_components(vals['C'], vals['L'])

    def set_Z_input(self, new_Z):
        if self.Z_input != new_Z:
            self.Z_input = new_Z
            self.Rin.set_resistance(new_Z)

    def set_Z_output(self, new_Z):
        if self.Z_output != new_Z:
            self.Z_output = new_Z
            self.Rt.set_resistance(new_Z)


mk2 = RCA_MK2_SEF(44100, 175, 3990)

colors = cm.rainbow(np.linspace(0,1,len(mk2.LP_vals)))

# plot HP cutoff positions
for i, fc in enumerate(mk2.HP_vals):
    mk2.set_highpass_cutoff(fc)
    plot_magnitude_response(mk2, label = f'{fc} hz', c = colors[i], title= 'HPF sweep ')

plt.show()

mk2.set_highpass_cutoff(170)

# plot LP cutoff positions
for i, fc in enumerate(mk2.LP_vals):
    mk2.set_lowpass_cutoff(fc)
    plot_magnitude_response(mk2, label = f'{fc} hz', c = colors[i], title= 'LPF sweep ')

plt.show()
