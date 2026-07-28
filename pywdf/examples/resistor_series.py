import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.wdf import *
from core.circuit import Circuit

import matplotlib.pyplot as plt


class ResistorSeries(Circuit):
    
    def __init__(
            self, 
            fs: int, 
            R1_val: float, 
            R2_val: float
        ) -> None:
        
        self.fs = fs

        self.R1 = Resistor(R1_val)
        self.R2 = Resistor(R2_val)
        self.S1 = SeriesAdaptor(self.R1, self.R2)
        # self.I1 = PolarityInverter(self.S1)  
        self.Vs = IdealVoltageSource(self.S1)

        super().__init__(self.Vs, self.Vs, self.R2)

    def set_R1(self,new_R):
        self.R1.set_resistance(new_R)

    def set_R2(self,new_R):
        self.R2.set_resistance(new_R)


if __name__ == '__main__':
    
    rs = ResistorSeries(44100, 1e4, 1e4)

    vs = np.arange(0.0, 6.0, 0.01)

    y = rs.process_i_v_signals(vs)

    v, _, i  = zip(*y)

    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6.5))
    
    # ax.plot(vs, label="voltage across", alpha=0.75)
    ax.plot(v, label="voltage across", alpha=0.75)
    ax.set_xlabel("sample")
    ax.set_ylabel("v[n]")
    
    color = 'tab:blue'
    ax2 = ax.twinx() 
    ax2.plot(i, color=color, label="current through output", alpha=0.75)
    ax2.set_ylabel('i[n]', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    ax.set_title(loc="left", label="v[n] voltage across and i[n] current through resistor R2")
    ax.grid(True)
    ax.legend()  

    plt.savefig("./tests/resistor_series.png")
    plt.show()


