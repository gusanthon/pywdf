import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.wdf import *
from core.circuit import Circuit

class VoltageDivider(Circuit):
    def __init__(self, fs: int, R1_val: float, R2_val: float) -> None:
        self.fs = fs

        self.R1 = Resistor(R1_val)
        self.R2 = Resistor(R2_val)

        self.P1 = ParallelAdaptor(self.R1, self.R2)
        self.I1 = PolarityInverter(self.P1)
        self.Vs = IdealVoltageSource(self.I1)

        super().__init__(self.Vs, self.Vs, self.R1)

    def set_R1(self,new_R):
        self.R1.set_resistance(new_R)

    def set_R2(self,new_R):
        self.R2.set_resistance(new_R)

