import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.wdf import *
from core.circuit import Circuit

import matplotlib.pyplot as plt

class _Resistor(Circuit):

    def __init__(   
            self, 
            fs: int, 
            R1_val: float 
        ) -> None:
        
        self.fs = fs

        self.R1 = Resistor(R1_val)

        self.Vs = IdealVoltageSource(self.R1)

        super().__init__(self.Vs, self.Vs, self.R1)

    def set_R1(self, new_R):
        self.R1.set_resistance(new_R)

    # def process_sample(self, sample: float) -> float:
    #     return super().process_sample_i_v(sample)


if __name__ == '__main__':

    r = _Resistor(44100, 1e4)

    vs = np.arange(0.0, 6.0, 0.01)

    y = r.process_i_v_signals(vs)

    v, _, i  = zip(*y)

    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6.5))
    ax.plot(v, label="voltage across", alpha=0.75)
    ax.set_xlabel("sample")
    ax.set_ylabel("v[n]")
    color = 'tab:blue'
    ax2 = ax.twinx() 
    ax2.set_ylabel('i[n]', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.plot(i, color=color, label="current through output", alpha=0.75)

    ax.set_title(loc="left", label="v[n] voltage across and i[n] current through resistor R1 ")
    ax.grid(True)
    ax.legend()  

    plt.savefig("./tests/resistor.png")
    plt.show()


