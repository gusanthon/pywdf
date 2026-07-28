import matplotlib.pyplot as plt
from pathlib import Path
import sys

import numpy as np

script_path = Path(__file__).resolve()
src_dir = script_path.parent.parent

sys.path.append(str(src_dir))

from core.wdf import IdealCurrentSource, Inductor
from core.circuit import Circuit

class _Inductor(Circuit):
    
    def __init__(
        self,
        sample_rate: int,
        alpha: int
    ) -> None:
    
        self.L = 0.5                # 500 mH
        self.fs = sample_rate
        self.twopi = 2 * np.pi

        self.L1 = Inductor(self.L, self.fs, alpha)
        self.Is = IdealCurrentSource(self.L1)

        # init and set circuit
        super().__init__(self.Is, self.Is, self.L1)

    def process_sample_i_v(
            self, 
            sample: float
        ) -> tuple[float, float, float]:

        """Process an individual sample with this circuit.

        Note: not every circuit will follow this general pattern, in such cases users may want to overwrite this function. See example circuits

        Args:
            sample (float): incoming sample to process

        Returns:
            (i, v) I-V tupple: processed sample
        """
        self.Is.set_current(sample)
        self.Is.accept_incident_wave(self.L1.propagate_reflected_wave())
        self.L1.accept_incident_wave(self.Is.propagate_reflected_wave())

        return ( self.output.wave_to_voltage(), self.source.wave_to_current(), self.output.wave_to_current() ) 


if __name__ == "__main__":

    fs = 44100                          # Samples/s
    f = 500.0                           # 500 Hz
    duration = .02                      # s

    _inductor = _Inductor(fs, 0.5)      # 0.0 – Backward Euler; 0.* – Damping ; 1.0 – Bilinear Transform 

    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    sine = np.sin(2 * np.pi * f * t)  

    y = _inductor.process_i_v_signals(sine)

    v, _, i  = zip(*y)

    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6.5))
    ax.plot(v, color="r", label="voltage across", alpha=0.75)
    ax.set_xlabel("sample")
    ax.set_ylabel("v[n]")
    
    color = 'tab:blue'
    ax2 = ax.twinx() 
    ax2.set_ylabel('i[n]', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.plot(i, color=color, label="current through", alpha=0.75)

    ax.set_title(loc="left", label="v[n] voltage across and i[n] current through inductor L")
    ax.legend()  
    ax2.legend()  
    ax.grid(True)

    plt.savefig("../../tests/inductor.png")
    plt.show()
