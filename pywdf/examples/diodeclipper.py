import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from core.wdf import *
from core.circuit import Circuit


class DiodeClipper(Circuit):
    def __init__(
        self, sample_rate: int, cutoff: float = 1000, input_gain_db: float = 0,
        output_gain_db: float = 0, n_diodes: float = 2
    ) -> None:

        self.fs = sample_rate
        self.cutoff = cutoff
        self.input_gain = 10 ** (input_gain_db / 20)
        self.input_gain_db = input_gain_db
        self.output_gain = 10 ** (output_gain_db / 20)
        self.output_gain_db = output_gain_db

        self.C = 47e-9
        self.R = 1.0 / (2 * np.pi * self.C * self.cutoff)

        self.R1 = Resistor(self.R)
        self.Vs = ResistiveVoltageSource()

        self.S1 = SeriesAdaptor(self.Vs, self.R1)
        self.C1 = Capacitor(self.C, self.fs)

        self.P1 = ParallelAdaptor(self.S1, self.C1)
        self.Dp = DiodePair(self.P1, 2.52e-9, n_diodes=n_diodes)

        super().__init__(self.Vs, self.Dp, self.C1)

    def process_sample(self, sample: float) -> float:
        sample *= self.input_gain
        return -(super().process_sample(sample) * self.output_gain) ### ¡! phase inverted !¡

    def set_cutoff(self, new_cutoff: float) -> None:
        if self.cutoff != new_cutoff:
            self.cutoff = new_cutoff
            self.R = 1.0 / (2 * np.pi * self.C * self.cutoff)
            self.R1.set_resistance(self.R)

    def set_input_gain(self, gain_db: float) -> None:
        if self.input_gain_db != gain_db:
            self.input_gain = 10 ** (gain_db / 20)
            self.input_gain_db = gain_db

    def set_output_gain(self, gain_db: float) -> None:
        if self.output_gain_db != gain_db:
            self.output_gain = 10 ** (gain_db / 20)
            self.output_gain_db = gain_db

    def set_num_diodes(self, new_n_diodes: float) -> None:
        if self.Dp.n_diodes != new_n_diodes:
            self.Dp.set_diode_params(self.Dp.Is, self.Dp.Vt, new_n_diodes)

if __name__ == "__main__":

    dc = DiodeClipper(44100, cutoff=5000, input_gain_db=5)
    dc.AC_transient_analysis()
