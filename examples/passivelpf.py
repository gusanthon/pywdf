import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.wdf import *
from core.circuit import Circuit

class PassiveLPF(Circuit):
    def __init__(self, sample_rate: int, cutoff: float = 1000) -> None:
        self.fs = sample_rate
        self.cutoff = cutoff
        self.def_cutoff = cutoff

        self.Z = 800
        self.C = (1 / self.Z) / (2.0 * np.pi * cutoff)

        self.R1 = Resistor(10)
        self.R2 = Resistor(1e4)

        self.C1 = Capacitor(self.C, self.fs)
        self.C2 = Capacitor(self.C, self.fs)

        self.S1 = SeriesAdaptor(self.R2, self.C2)
        self.P1 = ParallelAdaptor(self.C1, self.S1)
        self.S2 = SeriesAdaptor(self.R1, self.P1)

        self.Vs = IdealVoltageSource(self.P1)

        super().__init__(self.Vs, self.Vs, self.C2)

    def set_cutoff(self, new_cutoff: float) -> None:
        if self.cutoff != new_cutoff:
            self.cutoff = new_cutoff
            self.C = (1.0 / self.Z) / (2 * np.pi * self.cutoff)
            self.C1.set_capacitance(self.C)
            self.C2.set_capacitance(self.C)

if __name__ == '__main__':
    lpf = PassiveLPF(44100, 1000)
    lpf.plot_freqz()