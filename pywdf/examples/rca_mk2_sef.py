#  As described in : 
#  https://dafx2020.mdw.ac.at/proceedings/papers/DAFx20in22_paper_39.pdf

import sys, os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from core.wdf import *
from core.rtype import *
from core.circuit import Circuit

class HighPassStage:

    def __init__(self, connection, fs = 44100, C_HP = 1e-6, L_HP = 1e-3, k = 560):

        self.C_HP = C_HP
        self.L_HP = L_HP
        self.fs = fs
        self.k = k

        self.C_HP2 = Capacitor(self.C_HP, self.fs)
        self.S4 = SeriesAdaptor(self.C_HP2, connection)
        self.L_HP1 = Inductor(self.L_HP, self.fs)

        self.P2 = ParallelAdaptor(self.L_HP1, self.S4)
        self.C_HP1 = Capacitor(self.C_HP, self.fs)

        self.S3 = SeriesAdaptor(self.C_HP1, self.P2)
        self.C_HPm2 = Capacitor(self.C_HP, self.fs)

        self.S2 = SeriesAdaptor(self.C_HPm2, self.S3)
        self.L_HPm = Inductor(self.L_HP, self.fs)

        self.P1 = ParallelAdaptor(self.L_HPm, self.S2)
        self.C_HPm1 = Capacitor(self.C_HP, self.fs)

        self.S0 = SeriesAdaptor(self.P1, self.C_HPm1)

    def set_components(self, C_HP, L_HP, HP_mod, k):

        self.C_HP1.set_capacitance(C_HP)
        self.C_HP2.set_capacitance(C_HP)
        self.L_HP1.set_inductance(L_HP)

        if HP_mod == False:
            wc = 1e-8
            C_HP = np.sqrt(2) / (self.k * wc)
            L_HP = (np.sqrt(2) * self.k) / wc

        self.C_HPm1.set_capacitance(C_HP)
        self.C_HPm2.set_capacitance(C_HP)
        self.L_HPm.set_inductance(L_HP)


class LowPassStage:
    def __init__(self, connection, fs, C_LP, L_LP, k):

        self.fs = fs
        self.L_LP = L_LP
        self.C_LP = C_LP
        self.k = k

        self.L_LPm2 = Inductor(self.L_LP, self.fs)

        self.S8 = SeriesAdaptor(self.L_LPm2, connection)
        self.C_LPm1 = Capacitor(self.C_LP, self.fs)

        self.P4 = ParallelAdaptor(self.C_LPm1, self.S8)
        self.L_LPm1 = Inductor(self.L_LP, self.fs)

        self.S7 = SeriesAdaptor(self.L_LPm1, self.P4)
        self.L_LP2 = Inductor(self.L_LP, self.fs)

        self.S6 = SeriesAdaptor(self.L_LP2, self.S7)
        self.C_LP1 = Capacitor(self.C_LP, self.fs)

        self.P3 = ParallelAdaptor(self.C_LP1, self.S6)
        self.L_LP1 = Inductor(self.L_LP, self.fs)
        self.S5 = SeriesAdaptor(self.L_LP1, self.P3)

    def set_sample_rate(self, fs):
        self.fs = fs
        
        self.C_LP1.set_sample_rate(fs)
        self.L_LP1.set_sample_rate(fs)
        self.L_LP2.set_sample_rate(fs)

        self.C_LPm1.set_sample_rate(fs)
        self.L_LPm1.set_sample_rate(fs)
        self.L_LPm2.set_sample_rate(fs)

    def set_components(self, C_LP, L_LP, LP_mod, k):
        self.C_LP1.set_capacitance(C_LP)
        self.L_LP1.set_inductance(L_LP)
        self.L_LP2.set_inductance(L_LP)

        if LP_mod == False:
            wc = 1e8
            self.C_LP = (2 * np.sqrt(2)) / (self.k * wc)
            self.L_LP = (np.sqrt(2) * self.k) / wc

        self.C_LPm1.set_capacitance(C_LP)
        self.L_LPm1.set_inductance(L_LP)
        self.L_LPm2.set_inductance(L_LP)


class RCA_MK2_SEF(Circuit):
    
    def __init__(self, sample_rate: int, highpass_cutoff: float, lowpass_cutoff: float) -> None:

        # mod switches disabled by default

        self.lowpass_mod = False
        self.highpass_mod = False

        self.k = 560

        # measured C & L vals for each HP & LP knob position

        self.HP_vals = {
            0 : {'C' : 9999999, 'L' : 9999999},
            175 : {'C' : 1.6e-6, 'L' : 255.6e-3},
            248 : {'C' : 1.15e-6, 'L' : 176.9e-3},
            352 : {'C' : .8e-6, 'L' : 126.4e-3},
            497 : {'C' : .57e-6, 'L' : 90.11e-3},
            699 : {'C' : .4e-6, 'L' : 63.9e-3},
            1002 : {'C' : .272e-6, 'L' : 44.56e-3},
            1411 : {'C' : .2e-6, 'L' : 31.79e-3 },
            2024 : {'C' : .15e-6, 'L' : 21.77e-3},
            2847 : {'C' : .1e-6, 'L' : 15.63e-3 },
            3994 : {'C' : .069e-6, 'L' : 11.18e-3}
        }

        self.LP_vals = {
            175 : {'C' : 3.22e-6, 'L' : 511.1e-3 },
            245 : {'C' : 2.3e-6, 'L' :  365.2e-3 },
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

        self.highpass_cutoff = highpass_cutoff
        self.lowpass_cutoff = lowpass_cutoff

        self.C_HP = 1e-6
        self.L_HP = 1e-3
        self.C_LP = 1e-6
        self.L_LP = 1e-3

        self.Rt = Resistor(self.Z_output)

        self.num_HP_stages = 1
        self.num_LP_stages = 1 

        self.build_circuit()


    def build_circuit(self): 

        # LOW PASS STAGES
        first_stage = LowPassStage(self.Rt, self.fs, self.C_LP, self.L_LP, self.k)
        self.LP_stages = [first_stage]

        for i in range(1, self.num_LP_stages):
            connection = self.LP_stages[i - 1].S5
            stage = LowPassStage(connection, self.fs, self.C_LP, self.L_LP, self.k)
            self.LP_stages.append(stage)
        
        self.S8 = SeriesAdaptor(self.LP_stages[-1].L_LPm2, self.Rt) 

        # HIGH PASS STAGES
        connection = self.LP_stages[-1].S5
        first_stage = HighPassStage(connection, fs = self.fs, C_HP = self.C_HP, L_HP = self.L_HP, k = self.k)
        self.HP_stages = [first_stage]

        for i in range(1, self.num_HP_stages):
            connection = self.HP_stages[i - 1].S0
            stage = HighPassStage(connection, fs = self.fs, C_HP = self.C_HP, L_HP = self.L_HP, k = self.k)
            self.HP_stages.append(stage)

        # INPUT STAGE
        self.S1 = SeriesAdaptor(self.HP_stages[-1].C_HPm1, self.HP_stages[-1].P1)
        self.Rin = Resistor(self.Z_input)

        self.S0 = SeriesAdaptor(self.Rin, self.S1)
        self.Vin = IdealVoltageSource(self.S0)

        self.set_HP_components()
        self.set_LP_components()

        super().__init__(self.Vin, self.Vin, self.Rt)



    def set_num_LP_stages(self, nstages):
        assert nstages > 0
        self.num_LP_stages = nstages
        self.build_circuit()

    def set_num_HP_stages(self, nstages):
        assert nstages > 0
        self.num_HP_stages = nstages
        self.build_circuit()


    def set_HP_components(self):
        wc = self.highpass_cutoff * 2. * np.pi
        self.C_HP = np.sqrt(2) / (self.k * wc)
        self.L_HP = self.k / (2. * np.sqrt(2) * wc)

        for stage in self.HP_stages:
            stage.set_components(self.C_HP, self.L_HP, self.highpass_mod, self.k)

        
    def set_LP_components(self):
        wc = self.lowpass_cutoff * 2 * np.pi
        self.C_LP = (2 * np.sqrt(2)) / (self.k * wc)
        self.L_LP = (np.sqrt(2) * self.k) / wc

        for stage in self.LP_stages:
            stage.set_components(self.C_LP, self.L_LP, self.lowpass_mod, self.k)

    def set_lowpass_knob_position(self, pos):
        assert pos >= 0 and pos < len(self.LP_vals)

        self.lowpass_cutoff, vals = list(self.LP_vals.items())[pos]
        self.C_LP = vals['C']
        self.L_LP = vals['L']

        for stage in self.LP_stages:
            stage.set_components(self.C_LP, self.L_LP, self.lowpass_mod, self.k)

    def set_highpass_knob_position(self,pos):
        assert pos >= 0 and pos < len(self.HP_vals)

        self.highpass_cutoff, vals = list(self.HP_vals.items())[pos]        
        self.C_HP = vals['C']
        self.L_HP = vals['L']

        for stage in self.HP_stages:
            stage.set_components(self.C_HP, self.L_HP, self.highpass_mod, self.k)

    def process_sample(self, sample: float) -> float:
        gain_db = 6 # factor to compensate for gain loss
        k = 10 ** (gain_db / 20)
        return k * super().process_sample(sample)

    def set_highpass_cutoff(self, new_cutoff):
        self.highpass_cutoff = new_cutoff
        self.set_HP_components()

    def set_lowpass_cutoff(self, new_cutoff):
        self.lowpass_cutoff = new_cutoff
        self.set_LP_components()

    def set_highpass_mod(self, mod):
        if self.highpass_mod != mod:
            self.highpass_mod = mod
            self.set_HP_components()

    def set_lowpass_mod(self, mod):
        if self.set_lowpass_mod != mod:
            self.lowpass_mod = mod
            self.set_LP_components()

    def set_Z_input(self, new_Z):
        if self.Z_input != new_Z:
            self.Z_input = new_Z
            self.Rin.set_resistance(new_Z)

    def set_Z_output(self, new_Z):
        if self.Z_output != new_Z:
            self.Z_output = new_Z
            self.Rt.set_resistance(new_Z)



if __name__ == '__main__':

    mk2 = RCA_MK2_SEF(44100, 20, 20e3)

    zs = range(1000, 50000, 5000)
    mk2.plot_freqz_list(range(1,10), mk2.set_num_LP_stages, "num LP stages")

