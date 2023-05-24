"""
Passive All Pass Filter implementation based on:

Herencsar, N., Koton, J., Vrba, K., Minaei, S., & Göknar, I. C. (2015, August).
Voltage-mode all-pass filter passive scheme based on floating negative resistor
and grounded capacitor. In 2015 European Conference on Circuit Theory and Design
(ECCTD) (pp. 1-4). IEEE.
"""

from pathlib import Path
import sys

script_dir = Path(__file__).resolve().parent
wdf_dir = script_dir.parent
plot_dir = wdf_dir.parent / "data" / "plot"

# add path to PYTHONPATH
sys.path.append(str(wdf_dir))

from core.wdf import (
    Resistor,
    IdealVoltageSource,
    Capacitor,
    PolarityInverter,
)
from core.circuit import Circuit
from core.rtype import RTypeAdaptor


class PassiveAPF(Circuit):
    def __init__(self, sample_rate: int, cutoff: float = 1000) -> None:

        # define wdf params
        self.fs = sample_rate
        self.cutoff = cutoff

        # define component values
        self.R1_value = -10.8e3
        self.R2_value = 3.6e3
        self.R3_value = 5.4e3
        self.C1_value = 9.259e-8

        ## define ports

        # Port B
        self.R1 = Resistor(self.R1_value)

        # Port C
        self.R2 = Resistor(self.R2_value)

        # Port D
        self.R3 = Resistor(self.R3_value)

        # Port E
        self.C1 = Capacitor(self.C1_value, self.fs)

        # define R-TypeAdaptor
        self.R_adaptor = PolarityInverter(
            RTypeAdaptor([self.R1, self.R2, self.R3, self.C1], self.impedance_calc, 0)
        )

        self.Vin = IdealVoltageSource(self.R_adaptor)

        super().__init__(self.Vin, self.Vin, None)

    def impedance_calc(self, R: RTypeAdaptor):
        Rb, Rc, Rd, Re = R.get_port_impedances()
        R.set_S_matrix(
            [
                [
                    0,
                    -Rc / (Rb + Rc + Rd),
                    -(Rb + Rd) / (Rb + Rc + Rd),
                    -Rc / (Rb + Rc + Rd),
                    -1,
                ],
                [
                    -Rb * Rc / (Rb * Rc + Rc * Rd + (Rb + Rc + Rd) * Re),
                    -(
                        Rb * Rb * Rc
                        - Rc * Rc * Rd
                        - Rc * Rd * Rd
                        + (Rb * Rb - Rc * Rc - 2 * Rc * Rd - Rd * Rd) * Re
                    )
                    / (
                        Rb * Rb * Rc
                        + Rb * Rc * Rc
                        + Rc * Rd * Rd
                        + (2 * Rb * Rc + Rc * Rc) * Rd
                        + (
                            Rb * Rb
                            + 2 * Rb * Rc
                            + Rc * Rc
                            + 2 * (Rb + Rc) * Rd
                            + Rd * Rd
                        )
                        * Re
                    ),
                    (
                        Rb * Rb * Rc
                        + Rb * Rc * Rd
                        + 2 * (Rb * Rb + Rb * Rc + Rb * Rd) * Re
                    )
                    / (
                        Rb * Rb * Rc
                        + Rb * Rc * Rc
                        + Rc * Rd * Rd
                        + (2 * Rb * Rc + Rc * Rc) * Rd
                        + (
                            Rb * Rb
                            + 2 * Rb * Rc
                            + Rc * Rc
                            + 2 * (Rb + Rc) * Rd
                            + Rd * Rd
                        )
                        * Re
                    ),
                    -(
                        2 * Rb * Rb * Rc
                        + Rb * Rc * Rc
                        + 2 * Rb * Rc * Rd
                        + 2 * (Rb * Rb + Rb * Rc + Rb * Rd) * Re
                    )
                    / (
                        Rb * Rb * Rc
                        + Rb * Rc * Rc
                        + Rc * Rd * Rd
                        + (2 * Rb * Rc + Rc * Rc) * Rd
                        + (
                            Rb * Rb
                            + 2 * Rb * Rc
                            + Rc * Rc
                            + 2 * (Rb + Rc) * Rd
                            + Rd * Rd
                        )
                        * Re
                    ),
                    -Rb * Rc / (Rb * Rc + Rc * Rd + (Rb + Rc + Rd) * Re),
                ],
                [
                    -(Rb * Rc + Rc * Rd) / (Rb * Rc + Rc * Rd + (Rb + Rc + Rd) * Re),
                    (
                        Rb * Rc * Rc
                        + Rc * Rc * Rd
                        + 2 * (Rb * Rc + Rc * Rc + Rc * Rd) * Re
                    )
                    / (
                        Rb * Rb * Rc
                        + Rb * Rc * Rc
                        + Rc * Rd * Rd
                        + (2 * Rb * Rc + Rc * Rc) * Rd
                        + (
                            Rb * Rb
                            + 2 * Rb * Rc
                            + Rc * Rc
                            + 2 * (Rb + Rc) * Rd
                            + Rd * Rd
                        )
                        * Re
                    ),
                    -(
                        Rb * Rc * Rc
                        + Rc * Rc * Rd
                        - (Rb * Rb - Rc * Rc + 2 * Rb * Rd + Rd * Rd) * Re
                    )
                    / (
                        Rb * Rb * Rc
                        + Rb * Rc * Rc
                        + Rc * Rd * Rd
                        + (2 * Rb * Rc + Rc * Rc) * Rd
                        + (
                            Rb * Rb
                            + 2 * Rb * Rc
                            + Rc * Rc
                            + 2 * (Rb + Rc) * Rd
                            + Rd * Rd
                        )
                        * Re
                    ),
                    (
                        Rb * Rc * Rc
                        + Rc * Rc * Rd
                        + 2 * (Rb * Rc + Rc * Rc + Rc * Rd) * Re
                    )
                    / (
                        Rb * Rb * Rc
                        + Rb * Rc * Rc
                        + Rc * Rd * Rd
                        + (2 * Rb * Rc + Rc * Rc) * Rd
                        + (
                            Rb * Rb
                            + 2 * Rb * Rc
                            + Rc * Rc
                            + 2 * (Rb + Rc) * Rd
                            + Rd * Rd
                        )
                        * Re
                    ),
                    -(Rb * Rc + Rc * Rd) / (Rb * Rc + Rc * Rd + (Rb + Rc + Rd) * Re),
                ],
                [
                    -Rc * Rd / (Rb * Rc + Rc * Rd + (Rb + Rc + Rd) * Re),
                    -(
                        2 * Rc * Rd * Rd
                        + (2 * Rb * Rc + Rc * Rc) * Rd
                        + 2 * ((Rb + Rc) * Rd + Rd * Rd) * Re
                    )
                    / (
                        Rb * Rb * Rc
                        + Rb * Rc * Rc
                        + Rc * Rd * Rd
                        + (2 * Rb * Rc + Rc * Rc) * Rd
                        + (
                            Rb * Rb
                            + 2 * Rb * Rc
                            + Rc * Rc
                            + 2 * (Rb + Rc) * Rd
                            + Rd * Rd
                        )
                        * Re
                    ),
                    (Rb * Rc * Rd + Rc * Rd * Rd + 2 * ((Rb + Rc) * Rd + Rd * Rd) * Re)
                    / (
                        Rb * Rb * Rc
                        + Rb * Rc * Rc
                        + Rc * Rd * Rd
                        + (2 * Rb * Rc + Rc * Rc) * Rd
                        + (
                            Rb * Rb
                            + 2 * Rb * Rc
                            + Rc * Rc
                            + 2 * (Rb + Rc) * Rd
                            + Rd * Rd
                        )
                        * Re
                    ),
                    (
                        Rb * Rb * Rc
                        + Rb * Rc * Rc
                        - Rc * Rd * Rd
                        + (Rb * Rb + 2 * Rb * Rc + Rc * Rc - Rd * Rd) * Re
                    )
                    / (
                        Rb * Rb * Rc
                        + Rb * Rc * Rc
                        + Rc * Rd * Rd
                        + (2 * Rb * Rc + Rc * Rc) * Rd
                        + (
                            Rb * Rb
                            + 2 * Rb * Rc
                            + Rc * Rc
                            + 2 * (Rb + Rc) * Rd
                            + Rd * Rd
                        )
                        * Re
                    ),
                    -Rc * Rd / (Rb * Rc + Rc * Rd + (Rb + Rc + Rd) * Re),
                ],
                [
                    -(Rb + Rc + Rd) * Re / (Rb * Rc + Rc * Rd + (Rb + Rc + Rd) * Re),
                    -Rc * Re / (Rb * Rc + Rc * Rd + (Rb + Rc + Rd) * Re),
                    -(Rb + Rd) * Re / (Rb * Rc + Rc * Rd + (Rb + Rc + Rd) * Re),
                    -Rc * Re / (Rb * Rc + Rc * Rd + (Rb + Rc + Rd) * Re),
                    (Rb * Rc + Rc * Rd) / (Rb * Rc + Rc * Rd + (Rb + Rc + Rd) * Re),
                ],
            ]
        )
        Ra = (Rb * Rc + Rc * Rd + (Rb + Rc + Rd) * Re) / (Rb + Rc + Rd)
        return Ra

    def process_sample(self, sample):
        self.source.set_voltage(sample)
        self.root.accept_incident_wave(self.root.next.propagate_reflected_wave())
        self.root.next.accept_incident_wave(self.root.propagate_reflected_wave())
        return self.R3.wave_to_voltage() + self.C1.wave_to_voltage()

    def set_cutoff(self, cutoff: float) -> None:
        if self.cutoff != cutoff:
            self.cutoff = cutoff
            self.C1_value = 1.0 / (-self.R1.Rp * cutoff)
            print(self.C1_value)
            self.C1.set_capacitance(self.C1_value)


if __name__ == "__main__":
    apf = PassiveAPF(48000)
    apf.plot_freqz()
    apf.plot_freqz_list(
        [100, 250, 500, 1000, 2000, 4000, 8000], apf.set_cutoff, param_label="cutoff", outpath=plot_dir / "passive_apf-cutoff_list.png"
    )
