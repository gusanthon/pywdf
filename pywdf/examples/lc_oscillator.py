import matplotlib.pyplot as plt
from pathlib import Path
import sys

script_path = Path(__file__).resolve()
src_dir = script_path.parent.parent

sys.path.append(str(src_dir))

from core.wdf import *
from core.circuit import Circuit


class LCOscillator(Circuit):
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
        self.C = 1.0e-6
        self.L = 1.0e3
        self.twopi = 2 * np.pi

        self.C1 = Capacitor(self.C, self.fs)
        self.L1 = Inductor(self.L, self.fs)
        self.S2 = SeriesAdaptor(self.C1, self.L1)

        self.Vs = ResistiveVoltageSource()
        self.S1 = SeriesAdaptor(self.Vs, self.S2)

        self.SW1 = Switch(self.S1)

        # init and set circuit
        super().__init__(self.Vs, self.SW1, self.C1)
        self.set_params(self.frequency, self.closed, self.decibels)

    def process_sample(self, sample: float) -> float:
        return super().process_sample(sample) * self.gain

    def set_params(
        self, frequency: float, switch_closed: bool, decibels: float
    ) -> None:

        # update frequency
        if self.frequency != frequency:
            self.frequency = frequency

            self.L = 1.0 / (np.square(self.twopi * frequency) * self.C)
            self.L1.set_inductance(self.L)

        # update switch status
        if switch_closed != self.closed:
            self.SW1.set_closed(switch_closed)

        # update gain
        if self.decibels != decibels:
            self.decibels = decibels
            self.gain = self.decibels_to_gain()

    def decibels_to_gain(self):
        return 10 ** (self.decibels / 20.0)


if __name__ == "__main__":

    # set params
    fs = 48e3
    frequency = 440
    decibels = 0
    switch_closed = True

    lc_oscillator = LCOscillator(fs)
    lc_oscillator.set_params(frequency, switch_closed, decibels)  # update params

    # plot transfer function
    plt_dir = src_dir.parent / "data" / "plot"
    plt_dir.mkdir(exist_ok=True, parents=True)
    out_path = plt_dir / f"{script_path.stem}_{frequency}Hz.png"
    lc_oscillator.plot_freqz(out_path)

    # generate sinusoid
    out = lc_oscillator.process_signal(np.ones((int(fs))))
    out = out - 1  # remove DC offset [0, 2]

    plt.figure(figsize=(10, 4))
    plt.plot(out)
    plt.xlim([0, fs])
    plt.title(f"{frequency}Hz sinewave")
    plt.tight_layout()
    plt.show()

    # estimate pitch using zerocrossing
    zero_crossings = np.where(np.diff(np.sign(out)))[0]
    T = (zero_crossings[2] - zero_crossings[0]) / fs
    f = 1 / T
    print(f"T = {T:.6f}[s]\nf = {f:.3f}[Hz]")
