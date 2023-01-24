import sys
import os
import schemdraw
import schemdraw.elements as e

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.wdf import *
from core.circuit import Circuit

class VoltageDivider(Circuit):
    def __init__(self, fs: int, R1_val: float, R2_val: float) -> None:
        self.fs = fs

        self.R1 = Resistor(R1_val)
        self.R2 = Resistor(R2_val)

        self.S1 = SeriesAdaptor(self.R1, self.R2)
        self.Vs = IdealVoltageSource(self.S1)

        super().__init__(self.Vs, self.Vs, self.R1)

    def set_R1(self,new_R):
        self.R1.set_resistance(new_R)

    def set_R2(self,new_R):
        self.R2.set_resistance(new_R)

    def show_schematic(self, outpath=None):
        d = schemdraw.Drawing(fontsize=12)
        d.add(e.DOT, label='Input')

        R1 = d.add(e.RES, d='right', label='R1')
        R2 = d.add(e.RES, d='down', label='R2')
        d.add(e.GND)

        d.add(e.LINE, d='right', l=1, xy = R1.end)
        d.add(e.DOT, label='Output')
        d.draw()
        if outpath:
            d.save(outpath)

if __name__ == '__main__':
    vd = VoltageDivider(44100, 1e12, 1e10)
    # vd.plot_freqz()
    # vd.show_schematic()
    vd.AC_transient_analysis()