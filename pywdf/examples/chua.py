# Chua's Circuit example using WDF following the principles of Meerkotter and Scholz 1989.
# "Digital simulation of nonlinear circuits by wave digital filter principles"


import matplotlib.pyplot as plt
from pathlib import Path
import sys

script_path = Path(__file__).resolve()
src_dir = script_path.parent.parent

sys.path.append(str(src_dir))

from core.wdf import *
from core.circuit import Circuit


class Chua(Circuit):

    def __init__(
        self,
        sample_rate: int,
        frequency: float = 442,
        decibels: float = -18,
        closed: bool = True,
    ) -> None:

        self.fs = sample_rate
        self.frequency = frequency
        self.decibels = decibels
        self.gain = self.decibels_to_gain()
        self.closed = closed

        # initialize wdf
        self.C1_value = 5.5e-9
        self.R2_value = 1.3e3
        # self.R2_value = 1.428e3
        self.L3_value = 7.07e-3
        self.C4_value = 49.5e-9

        self.C1 = Capacitor(self.C1_value, self.fs)    # fs, needs local discretization
        self.R2 = Resistor(self.R2_value)              
        self.L3 = Inductor(self.L3_value, self.fs)     # fs, needs local discretization
        self.C4 = Capacitor(self.C4_value, self.fs)    # fs, needs local discretization

        self.Vs = SeriesVoltage(self.C4)

        self.P2 = ParallelAdaptor(self.L3, self.Vs)
        self.S1 = SeriesAdaptor(self.P2, self.R2)
        self.P1 = ParallelAdaptor(self.S1, self.C1)

        self.g1 = -500.0e-6 
        self.g2 = -800.0e-6
        self.v0 = 1.0
        self.R_NL = 569.2

        self.NL = ChuaDiode(
            self.P1, 
            g1=self.g1,         # parameter 1
            g2=self.g2,         # parameter 2
            v0=self.v0,         # Voltage parameter
            r1=self.R_NL,       # Resistance in Ohms
        )     
        
        # init and set circuit
        super().__init__(self.Vs, self.NL, self.C1)


    def process_sample(
        self, 
        sample: float
    ) -> float:
    
        self.Vs.set_voltage(sample)
        self.NL.accept_incident_wave(self.P1.propagate_reflected_wave())
        self.P1.accept_incident_wave(self.NL.propagate_reflected_wave())

        return self.output.wave_to_voltage()


    def process_sample_chua(self, sample: float) -> float:
        """Process an individual sample with this circuit.

        Note: not every circuit will follow this general pattern, in such cases users may want to overwrite this function. See example circuits

        Args:
            sample (float): incoming sample to process

        Returns:
            (i, v) I-V tupple: processed sample
        """
        self.Vs.set_voltage(sample)
        self.NL.accept_incident_wave(self.P1.propagate_reflected_wave())
        self.P1.accept_incident_wave(self.NL.propagate_reflected_wave())

        return self.output.wave_to_voltage(), self.C4.wave_to_voltage(), self.L3.wave_to_current() 


    def set_params(
        self, 
        frequency: float,
        decibels: float
    ) -> None:

        # update frequency
        if self.frequency != frequency:
            self.frequency = frequency

            # self.L = 1.0 / (np.square(self.twopi * frequency) * self.C)
            # self.L1.set_inductance(self.L)

        # update gain
        if self.decibels != decibels:
            self.decibels = decibels
            self.gain = self.decibels_to_gain()

    def decibels_to_gain(self):
        return 10 ** (self.decibels / 20.0)


if __name__ == "__main__":

    # set params
    # fs = 48e3       # doesn't work for this sample rate
    # fs = 96e3
    fs = 100e3
   
    chua_circuit = Chua(fs)
    # chua_circuit.set_params(frequency, decibels)  # update params

    # plot transfer function
    plt_dir = src_dir.parent / "tests" / "plots"
    plt_dir.mkdir(exist_ok=True, parents=True)
    out_path = plt_dir / f"{script_path.stem}_{fs}Hz.png"
    

    a = np.ones(1)  # impulse signal
    b = np.zeros(2400) 
    samples = np.concatenate((a, b))
    
    i_list, v1_list, v2_list = [], [], []

    for s in samples:
        v1, v2, i = chua_circuit.process_sample_chua(s)
        v1_list.append(v1)
        v2_list.append(v2)
        i_list.append(i)

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(v1_list, label='C1 [V]', color='r')
    axes[1].plot(v2_list, label='C4 [V]', color='g')
    axes[2].plot(i_list, label='L3 [I]', color='b')


    for ax in axes:
        ax.legend()
        ax.grid(True)
        ax.set_xlabel("Sample [n]")

    axes[0].set_ylabel("Output")

    # Centered title
    fig.suptitle("Chua impulse response", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle

    # Save and show
    out_path = plt_dir / f"{script_path.stem}_signal.png"
    plt.savefig(out_path.with_suffix('.png'))
    plt.show()


    # 3D Phase Portrait
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(v1_list, v2_list, i_list, lw=1.0)
    ax.set_title("3D Phase Portrait of Chua's Circuit WDF")
    ax.set_xlabel("x C1 [V]")
    ax.set_ylabel("y C4 [V]")
    ax.set_zlabel("z L3 [I]")
    plt.tight_layout()
    out_path = plt_dir / f"{script_path.stem}_WDF_attractor.png"
    plt.savefig(out_path.with_suffix('.png'))
    plt.show()
