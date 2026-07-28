import sys
import os
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.wdf import Capacitor, IdealVoltageSource
from core.circuit import Circuit

class _Capacitor(Circuit):

    def __init__(
            self, 
            sample_rate: int,
            tolerance: float,
            alpha: float,
        ) -> None:

        self.C  = 47e-9          # capacitance in Farads
        self.fs = sample_rate
        self.C1 = Capacitor(self.C, self.fs, tolerance, alpha)
        self.Vs = IdealVoltageSource(self.C1)

        super().__init__(self.Vs, self.Vs, self.C1)


    def process_sample_i_v(self, sample: float) -> float:
        """Process an individual sample with this circuit.

        Note: not every circuit will follow this general pattern, in such cases users may want to overwrite this function. See example circuits

        Args:
            sample (float): incoming sample to process

        Returns:
            (i, v) I-V tupple: processed sample
        """
        self.Vs.set_voltage(sample)
        self.Vs.accept_incident_wave(self.C1.propagate_reflected_wave())
        self.C1.accept_incident_wave(self.Vs.propagate_reflected_wave())

        return ( self.output.wave_to_voltage(), self.source.wave_to_current(), self.output.wave_to_current() ) 




if __name__ == "__main__":
    
    fs = 48_000
    capacitance = 47e-9
    frequency = 1_000.0
    amplitude = 1.0
    duration = 5e-3
    alpha = 0.0                         # Default 0.0 – Backward Euler 

    capacitor_circuit = _Capacitor(
        sample_rate = fs,
        tolerance   = 0.0,
        alpha       = alpha,            # 0.0 – Backward Euler; 1.0 – Bilinear Transform 
    )

 
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    time = np.arange(int(fs * duration)) / fs

    input_voltage = amplitude * np.sin(
        2.0 * np.pi * frequency * time
    )

    result = capacitor_circuit.process_i_v_signals(
        input_voltage
    )

    voltage, source_current, current = map(        
        np.asarray,
        zip(*result),
    )


    # Backward Euler reference:
    previous_voltage = np.concatenate(
        ([0.0], voltage[:-1])
    )
    expected_current = (
        capacitance
        * fs
        * (voltage - previous_voltage)
    )

    # np.testing.assert_allclose(   # Asserts for Backwards Euler
    #     voltage,
    #     input_voltage,
    #     rtol=1e-12,
    #     atol=1e-12,
    # )

    # np.testing.assert_allclose(
    #     current,
    #     expected_current,
    #     rtol=1e-10,
    #     atol=1e-14,
    # )

    _, ax_voltage = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(10, 6.5),
    )

    ax_voltage.plot(
        time * 1e3,
        voltage,
        label="Capacitor voltage",
    )
    ax_voltage.set_xlabel("Time [ms]")
    ax_voltage.set_ylabel("Voltage [V]")
    ax_voltage.grid(True)

    ax_current = ax_voltage.twinx()

    ax_current.plot(
        time * 1e3,
        current * 1e6,
        label="WDF capacitor current",
        alpha=0.75,
    )

    ax_current.plot(
        time * 1e3,
        expected_current * 1e6,
        linestyle="--",
        label="Expected BE current",
        alpha=0.75,
    )

    ax_current.set_ylabel("Current [µA]")

    ax_voltage.set_title(
        "Voltage-driven 47 nF capacitor"
    )

    lines_1, labels_1 = (
        ax_voltage.get_legend_handles_labels()
    )
    lines_2, labels_2 = (
        ax_current.get_legend_handles_labels()
    )

    ax_voltage.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc="upper right",
    )

    plt.tight_layout()
    plt.savefig(
        "../../tests/capacitor.png",
        dpi=150,
    )
    plt.show()