import sys
from pathlib import Path


script_path = Path(__file__).resolve()
src_dir = script_path.parent.parent
sys.path.append(str(src_dir))

from core.wdf import *
from core.rtype import *
from core.circuit import Circuit


def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)


class SallenKeyFilter(Circuit):
    def __init__(self, sample_rate: int, cutoff: float, q_val: float) -> None:


        self.fs = sample_rate

        self.capVal = 1.0e-8
        self.capRatio = 22.0

        # Port B
        self.R1 = Resistor(1e6) 

        # Port C
        self.C2 = Capacitor(self.capVal * self.capRatio, self.fs)

        # Port D
        self.R2 = Resistor(1e3) 
        self.R_adaptor = RTypeAdaptor([self.R1, self.C2, self.R2], self.__impedance_calc, 0)

        # Port A
        self.C1 = Capacitor(self.capVal / self.capRatio, self.fs) 
        self.S1 = SeriesAdaptor(self.R_adaptor, self.C1)
        self.Vin = IdealVoltageSource(self.S1)

        super().__init__(self.Vin, self.Vin, self.Vin)

    def process_sample(self, sample):
        self.source.set_voltage(sample)
        self.root.accept_incident_wave(self.root.next.propagate_reflected_wave())
        self.root.next.accept_incident_wave(self.root.propagate_reflected_wave())
        return self.R2.wave_to_voltage() # -310dB and flat phase

    def __impedance_calc(self, R: RTypeAdaptor) -> float:
        Ag = 100.0
        Ri = 1.0e9
        Ro = 1.0e-1

        Rb, Rc, Rd = R.get_port_impedances()
        print(Rb, Rc, Rd)

        R.set_S_matrix ([[ 0, -(Rc * Rd + ((Ag + 1) * Rc + (Ag + 1) * Rd) * Ri - Rc * Ro) / ((Rb + Rc) * Rd + ((Ag + 1) * Rb + (Ag + 1) * Rc + Rd) * Ri - (Rb + Rc + Ri) * Ro), -(Rb * Rd + ((Ag + 1) * Rb - Ag * Rd) * Ri - (Rb + Ri) * Ro) / ((Rb + Rc) * Rd + ((Ag + 1) * Rb + (Ag + 1) * Rc + Rd) * Ri - (Rb + Rc + Ri) * Ro), (((Ag + 1) * Rb + Ag * Rc) * Ri - (Rb + Rc + Ri) * Ro) / ((Rb + Rc) * Rd + ((Ag + 1) * Rb + (Ag + 1) * Rc + Rd) * Ri - (Rb + Rc + Ri) * Ro) ],
                                [ -(Rb * Rc * Rd - Rb * Rc * Ro + ((Ag + 1) * Rb * Rc + Rb * Rd) * Ri) / (Rb * Rc * Rd + ((Ag + 1) * Rb * Rc + (Ag + 1) * Rb * Rd) * Ri - (Rb * Rc + (Rb + Rc) * Rd + (Rc + Rd) * Ri) * Ro), -(Rb * Rb * Rc * Rd * Rd + ((Ag * Ag + 2 * Ag + 1) * Rb * Rb * Rc + (Ag * Ag + 2 * Ag + 1) * Rb * Rb * Rd) * Ri * Ri + (Rb * Rb * Rc - (Rc + Rd) * Ri * Ri + (Rb * Rb - Rc * Rc) * Rd - (Rc * Rc + 2 * Rc * Rd) * Ri) * Ro * Ro + (2 * (Ag + 1) * Rb * Rb * Rc * Rd + (Ag + 1) * Rb * Rb * Rd * Rd) * Ri - (2 * Rb * Rb * Rc * Rd + (Rb * Rb - Rc * Rc) * Rd * Rd - ((Ag + 1) * Rc * Rc + (Ag + 2) * Rc * Rd + Rd * Rd) * Ri * Ri + (2 * (Ag + 1) * Rb * Rb * Rc - 2 * Rc * Rd * Rd + (2 * (Ag + 1) * Rb * Rb - (Ag + 2) * Rc * Rc) * Rd) * Ri) * Ro) / ((Rb * Rb * Rc + Rb * Rc * Rc) * Rd * Rd + ((Ag * Ag + 2 * Ag + 1) * Rb * Rb * Rc + (Ag * Ag + 2 * Ag + 1) * Rb * Rc * Rc + (Ag + 1) * Rb * Rd * Rd + ((Ag * Ag + 2 * Ag + 1) * Rb * Rb + (Ag * Ag + 3 * Ag + 2) * Rb * Rc) * Rd) * Ri * Ri + (Rb * Rb * Rc + Rb * Rc * Rc + (Rc + Rd) * Ri * Ri + (Rb * Rb + 2 * Rb * Rc + Rc * Rc) * Rd + (2 * Rb * Rc + Rc * Rc + 2 * (Rb + Rc) * Rd) * Ri) * Ro * Ro + (((Ag + 1) * Rb * Rb + (Ag + 2) * Rb * Rc) * Rd * Rd + 2 * ((Ag + 1) * Rb * Rb * Rc + (Ag + 1) * Rb * Rc * Rc) * Rd) * Ri - ((Rb * Rb + 2 * Rb * Rc + Rc * Rc) * Rd * Rd + (2 * (Ag + 1) * Rb * Rc + (Ag + 1) * Rc * Rc + (2 * (Ag + 1) * Rb + (Ag + 2) * Rc) * Rd + Rd * Rd) * Ri * Ri + 2 * (Rb * Rb * Rc + Rb * Rc * Rc) * Rd + (2 * (Ag + 1) * Rb * Rb * Rc + 2 * (Ag + 1) * Rb * Rc * Rc + 2 * (Rb + Rc) * Rd * Rd + (2 * (Ag + 1) * Rb * Rb + 3 * (Ag + 2) * Rb * Rc + (Ag + 2) * Rc * Rc) * Rd) * Ri) * Ro), (Rb * Rb * Rc * Rd * Rd + ((Ag * Ag + 2 * Ag + 1) * Rb * Rb * Rc + Ag * Rb * Rd * Rd + ((2 * Ag * Ag + 3 * Ag + 1) * Rb * Rb + (Ag * Ag + Ag) * Rb * Rc) * Rd) * Ri * Ri + (Rb * Rb * Rc + 2 * (Rb * Rb + Rb * Rc) * Rd + (Rb * Rc + 2 * Rb * Rd) * Ri) * Ro * Ro + (2 * (Ag + 1) * Rb * Rb * Rc * Rd + ((2 * Ag + 1) * Rb * Rb + Ag * Rb * Rc) * Rd * Rd) * Ri - (2 * Rb * Rb * Rc * Rd + 2 * (Rb * Rb + Rb * Rc) * Rd * Rd + ((Ag + 1) * Rb * Rc + (2 * Ag + 1) * Rb * Rd) * Ri * Ri + (2 * (Ag + 1) * Rb * Rb * Rc + 2 * Rb * Rd * Rd + ((4 * Ag + 3) * Rb * Rb + 3 * (Ag + 1) * Rb * Rc) * Rd) * Ri) * Ro) / ((Rb * Rb * Rc + Rb * Rc * Rc) * Rd * Rd + ((Ag * Ag + 2 * Ag + 1) * Rb * Rb * Rc + (Ag * Ag + 2 * Ag + 1) * Rb * Rc * Rc + (Ag + 1) * Rb * Rd * Rd + ((Ag * Ag + 2 * Ag + 1) * Rb * Rb + (Ag * Ag + 3 * Ag + 2) * Rb * Rc) * Rd) * Ri * Ri + (Rb * Rb * Rc + Rb * Rc * Rc + (Rc + Rd) * Ri * Ri + (Rb * Rb + 2 * Rb * Rc + Rc * Rc) * Rd + (2 * Rb * Rc + Rc * Rc + 2 * (Rb + Rc) * Rd) * Ri) * Ro * Ro + (((Ag + 1) * Rb * Rb + (Ag + 2) * Rb * Rc) * Rd * Rd + 2 * ((Ag + 1) * Rb * Rb * Rc + (Ag + 1) * Rb * Rc * Rc) * Rd) * Ri - ((Rb * Rb + 2 * Rb * Rc + Rc * Rc) * Rd * Rd + (2 * (Ag + 1) * Rb * Rc + (Ag + 1) * Rc * Rc + (2 * (Ag + 1) * Rb + (Ag + 2) * Rc) * Rd + Rd * Rd) * Ri * Ri + 2 * (Rb * Rb * Rc + Rb * Rc * Rc) * Rd + (2 * (Ag + 1) * Rb * Rb * Rc + 2 * (Ag + 1) * Rb * Rc * Rc + 2 * (Rb + Rc) * Rd * Rd + (2 * (Ag + 1) * Rb * Rb + 3 * (Ag + 2) * Rb * Rc + (Ag + 2) * Rc * Rc) * Rd) * Ri) * Ro), (((Ag - 1) * Rb * Rb * Rc + Ag * Rb * Rc * Rc) * Rd * Ri + ((Ag * Ag - 1) * Rb * Rb * Rc + (Ag * Ag + Ag) * Rb * Rc * Rc - ((Ag + 1) * Rb * Rb - Ag * Rb * Rc) * Rd) * Ri * Ri + (Rb * Rb * Rc + Rb * Rc * Rc + Rb * Rc * Ri) * Ro * Ro - (((Ag - 1) * Rb * Rc - Rb * Rd) * Ri * Ri + (Rb * Rb * Rc + Rb * Rc * Rc) * Rd + (2 * Ag * Rb * Rb * Rc + (2 * Ag + 1) * Rb * Rc * Rc - Rb * Rb * Rd) * Ri) * Ro) / ((Rb * Rb * Rc + Rb * Rc * Rc) * Rd * Rd + ((Ag * Ag + 2 * Ag + 1) * Rb * Rb * Rc + (Ag * Ag + 2 * Ag + 1) * Rb * Rc * Rc + (Ag + 1) * Rb * Rd * Rd + ((Ag * Ag + 2 * Ag + 1) * Rb * Rb + (Ag * Ag + 3 * Ag + 2) * Rb * Rc) * Rd) * Ri * Ri + (Rb * Rb * Rc + Rb * Rc * Rc + (Rc + Rd) * Ri * Ri + (Rb * Rb + 2 * Rb * Rc + Rc * Rc) * Rd + (2 * Rb * Rc + Rc * Rc + 2 * (Rb + Rc) * Rd) * Ri) * Ro * Ro + (((Ag + 1) * Rb * Rb + (Ag + 2) * Rb * Rc) * Rd * Rd + 2 * ((Ag + 1) * Rb * Rb * Rc + (Ag + 1) * Rb * Rc * Rc) * Rd) * Ri - ((Rb * Rb + 2 * Rb * Rc + Rc * Rc) * Rd * Rd + (2 * (Ag + 1) * Rb * Rc + (Ag + 1) * Rc * Rc + (2 * (Ag + 1) * Rb + (Ag + 2) * Rc) * Rd + Rd * Rd) * Ri * Ri + 2 * (Rb * Rb * Rc + Rb * Rc * Rc) * Rd + (2 * (Ag + 1) * Rb * Rb * Rc + 2 * (Ag + 1) * Rb * Rc * Rc + 2 * (Rb + Rc) * Rd * Rd + (2 * (Ag + 1) * Rb * Rb + 3 * (Ag + 2) * Rb * Rc + (Ag + 2) * Rc * Rc) * Rd) * Ri) * Ro)],
                                [ -((Ag + 1) * Rb * Rc * Ri + Rb * Rc * Rd - (Rb * Rc + Rc * Ri) * Ro) / (Rb * Rc * Rd + ((Ag + 1) * Rb * Rc + (Ag + 1) * Rb * Rd) * Ri - (Rb * Rc + (Rb + Rc) * Rd + (Rc + Rd) * Ri) * Ro), (Rb * Rc * Rc * Rd * Rd + ((Ag * Ag + 2 * Ag + 1) * Rb * Rc * Rc + (Ag * Ag + 2 * Ag + 1) * Rb * Rc * Rd) * Ri * Ri + (Rb * Rc * Rc + 2 * (Rb * Rc + Rc * Rc) * Rd + (Rc * Rc + 2 * Rc * Rd) * Ri) * Ro * Ro + (2 * (Ag + 1) * Rb * Rc * Rc * Rd + (Ag + 1) * Rb * Rc * Rd * Rd) * Ri - (2 * Rb * Rc * Rc * Rd + 2 * (Rb * Rc + Rc * Rc) * Rd * Rd + ((Ag + 1) * Rc * Rc + (Ag + 1) * Rc * Rd) * Ri * Ri + (2 * (Ag + 1) * Rb * Rc * Rc + 2 * Rc * Rd * Rd + (3 * (Ag + 1) * Rb * Rc + (2 * Ag + 3) * Rc * Rc) * Rd) * Ri) * Ro) / ((Rb * Rb * Rc + Rb * Rc * Rc) * Rd * Rd + ((Ag * Ag + 2 * Ag + 1) * Rb * Rb * Rc + (Ag * Ag + 2 * Ag + 1) * Rb * Rc * Rc + (Ag + 1) * Rb * Rd * Rd + ((Ag * Ag + 2 * Ag + 1) * Rb * Rb + (Ag * Ag + 3 * Ag + 2) * Rb * Rc) * Rd) * Ri * Ri + (Rb * Rb * Rc + Rb * Rc * Rc + (Rc + Rd) * Ri * Ri + (Rb * Rb + 2 * Rb * Rc + Rc * Rc) * Rd + (2 * Rb * Rc + Rc * Rc + 2 * (Rb + Rc) * Rd) * Ri) * Ro * Ro + (((Ag + 1) * Rb * Rb + (Ag + 2) * Rb * Rc) * Rd * Rd + 2 * ((Ag + 1) * Rb * Rb * Rc + (Ag + 1) * Rb * Rc * Rc) * Rd) * Ri - ((Rb * Rb + 2 * Rb * Rc + Rc * Rc) * Rd * Rd + (2 * (Ag + 1) * Rb * Rc + (Ag + 1) * Rc * Rc + (2 * (Ag + 1) * Rb + (Ag + 2) * Rc) * Rd + Rd * Rd) * Ri * Ri + 2 * (Rb * Rb * Rc + Rb * Rc * Rc) * Rd + (2 * (Ag + 1) * Rb * Rb * Rc + 2 * (Ag + 1) * Rb * Rc * Rc + 2 * (Rb + Rc) * Rd * Rd + (2 * (Ag + 1) * Rb * Rb + 3 * (Ag + 2) * Rb * Rc + (Ag + 2) * Rc * Rc) * Rd) * Ri) * Ro), -(Rb * Rc * Rc * Rd * Rd + ((Ag * Ag + 2 * Ag + 1) * Rb * Rc * Rc - (Ag * Ag + 2 * Ag + 1) * Rb * Rb * Rd - (Ag + 1) * Rb * Rd * Rd) * Ri * Ri + (Rb * Rc * Rc - Rd * Ri * Ri - (Rb * Rb - Rc * Rc) * Rd + (Rc * Rc - 2 * Rb * Rd) * Ri) * Ro * Ro + (2 * (Ag + 1) * Rb * Rc * Rc * Rd - (Ag + 1) * Rb * Rb * Rd * Rd) * Ri - (2 * Rb * Rc * Rc * Rd - (Rb * Rb - Rc * Rc) * Rd * Rd + ((Ag + 1) * Rc * Rc - 2 * (Ag + 1) * Rb * Rd - Rd * Rd) * Ri * Ri + (2 * (Ag + 1) * Rb * Rc * Rc - 2 * Rb * Rd * Rd - (2 * (Ag + 1) * Rb * Rb - (Ag + 2) * Rc * Rc) * Rd) * Ri) * Ro) / ((Rb * Rb * Rc + Rb * Rc * Rc) * Rd * Rd + ((Ag * Ag + 2 * Ag + 1) * Rb * Rb * Rc + (Ag * Ag + 2 * Ag + 1) * Rb * Rc * Rc + (Ag + 1) * Rb * Rd * Rd + ((Ag * Ag + 2 * Ag + 1) * Rb * Rb + (Ag * Ag + 3 * Ag + 2) * Rb * Rc) * Rd) * Ri * Ri + (Rb * Rb * Rc + Rb * Rc * Rc + (Rc + Rd) * Ri * Ri + (Rb * Rb + 2 * Rb * Rc + Rc * Rc) * Rd + (2 * Rb * Rc + Rc * Rc + 2 * (Rb + Rc) * Rd) * Ri) * Ro * Ro + (((Ag + 1) * Rb * Rb + (Ag + 2) * Rb * Rc) * Rd * Rd + 2 * ((Ag + 1) * Rb * Rb * Rc + (Ag + 1) * Rb * Rc * Rc) * Rd) * Ri - ((Rb * Rb + 2 * Rb * Rc + Rc * Rc) * Rd * Rd + (2 * (Ag + 1) * Rb * Rc + (Ag + 1) * Rc * Rc + (2 * (Ag + 1) * Rb + (Ag + 2) * Rc) * Rd + Rd * Rd) * Ri * Ri + 2 * (Rb * Rb * Rc + Rb * Rc * Rc) * Rd + (2 * (Ag + 1) * Rb * Rb * Rc + 2 * (Ag + 1) * Rb * Rc * Rc + 2 * (Rb + Rc) * Rd * Rd + (2 * (Ag + 1) * Rb * Rb + 3 * (Ag + 2) * Rb * Rc + (Ag + 2) * Rc * Rc) * Rd) * Ri) * Ro), (((Ag + 1) * Rb * Rb * Rc + (Ag + 2) * Rb * Rc * Rc) * Rd * Ri + ((Ag * Ag + 2 * Ag + 1) * Rb * Rb * Rc + (Ag * Ag + 3 * Ag + 2) * Rb * Rc * Rc + 2 * (Ag + 1) * Rb * Rc * Rd) * Ri * Ri + (Rb * Rb * Rc + Rb * Rc * Rc + Rc * Ri * Ri + (2 * Rb * Rc + Rc * Rc) * Ri) * Ro * Ro - ((2 * (Ag + 1) * Rb * Rc + (Ag + 2) * Rc * Rc + 2 * Rc * Rd) * Ri * Ri + (Rb * Rb * Rc + Rb * Rc * Rc) * Rd + (2 * (Ag + 1) * Rb * Rb * Rc + (2 * Ag + 3) * Rb * Rc * Rc + (3 * Rb * Rc + 2 * Rc * Rc) * Rd) * Ri) * Ro) / ((Rb * Rb * Rc + Rb * Rc * Rc) * Rd * Rd + ((Ag * Ag + 2 * Ag + 1) * Rb * Rb * Rc + (Ag * Ag + 2 * Ag + 1) * Rb * Rc * Rc + (Ag + 1) * Rb * Rd * Rd + ((Ag * Ag + 2 * Ag + 1) * Rb * Rb + (Ag * Ag + 3 * Ag + 2) * Rb * Rc) * Rd) * Ri * Ri + (Rb * Rb * Rc + Rb * Rc * Rc + (Rc + Rd) * Ri * Ri + (Rb * Rb + 2 * Rb * Rc + Rc * Rc) * Rd + (2 * Rb * Rc + Rc * Rc + 2 * (Rb + Rc) * Rd) * Ri) * Ro * Ro + (((Ag + 1) * Rb * Rb + (Ag + 2) * Rb * Rc) * Rd * Rd + 2 * ((Ag + 1) * Rb * Rb * Rc + (Ag + 1) * Rb * Rc * Rc) * Rd) * Ri - ((Rb * Rb + 2 * Rb * Rc + Rc * Rc) * Rd * Rd + (2 * (Ag + 1) * Rb * Rc + (Ag + 1) * Rc * Rc + (2 * (Ag + 1) * Rb + (Ag + 2) * Rc) * Rd + Rd * Rd) * Ri * Ri + 2 * (Rb * Rb * Rc + Rb * Rc * Rc) * Rd + (2 * (Ag + 1) * Rb * Rb * Rc + 2 * (Ag + 1) * Rb * Rc * Rc + 2 * (Rb + Rc) * Rd * Rd + (2 * (Ag + 1) * Rb * Rb + 3 * (Ag + 2) * Rb * Rc + (Ag + 2) * Rc * Rc) * Rd) * Ri) * Ro)],
                                [ ((Ag + 1) * Rb * Rd * Ri - ((Rb + Rc) * Rd + Rd * Ri) * Ro) / (Rb * Rc * Rd + ((Ag + 1) * Rb * Rc + (Ag + 1) * Rb * Rd) * Ri - (Rb * Rc + (Rb + Rc) * Rd + (Rc + Rd) * Ri) * Ro), -((Ag + 1) * Rb * Rc * Rd * Rd * Ri + ((Ag * Ag + 2 * Ag + 1) * Rb * Rc * Rd + (Ag * Ag + 2 * Ag + 1) * Rb * Rd * Rd) * Ri * Ri - (Rc * Rd * Ri + (Rb * Rc + Rc * Rc) * Rd) * Ro * Ro + ((Rb * Rc + Rc * Rc) * Rd * Rd - ((Ag + 1) * Rc * Rd + (Ag + 1) * Rd * Rd) * Ri * Ri + ((Ag + 1) * Rc * Rc * Rd - ((Ag + 1) * Rb + Ag * Rc) * Rd * Rd) * Ri) * Ro) / ((Rb * Rb * Rc + Rb * Rc * Rc) * Rd * Rd + ((Ag * Ag + 2 * Ag + 1) * Rb * Rb * Rc + (Ag * Ag + 2 * Ag + 1) * Rb * Rc * Rc + (Ag + 1) * Rb * Rd * Rd + ((Ag * Ag + 2 * Ag + 1) * Rb * Rb + (Ag * Ag + 3 * Ag + 2) * Rb * Rc) * Rd) * Ri * Ri + (Rb * Rb * Rc + Rb * Rc * Rc + (Rc + Rd) * Ri * Ri + (Rb * Rb + 2 * Rb * Rc + Rc * Rc) * Rd + (2 * Rb * Rc + Rc * Rc + 2 * (Rb + Rc) * Rd) * Ri) * Ro * Ro + (((Ag + 1) * Rb * Rb + (Ag + 2) * Rb * Rc) * Rd * Rd + 2 * ((Ag + 1) * Rb * Rb * Rc + (Ag + 1) * Rb * Rc * Rc) * Rd) * Ri - ((Rb * Rb + 2 * Rb * Rc + Rc * Rc) * Rd * Rd + (2 * (Ag + 1) * Rb * Rc + (Ag + 1) * Rc * Rc + (2 * (Ag + 1) * Rb + (Ag + 2) * Rc) * Rd + Rd * Rd) * Ri * Ri + 2 * (Rb * Rb * Rc + Rb * Rc * Rc) * Rd + (2 * (Ag + 1) * Rb * Rb * Rc + 2 * (Ag + 1) * Rb * Rc * Rc + 2 * (Rb + Rc) * Rd * Rd + (2 * (Ag + 1) * Rb * Rb + 3 * (Ag + 2) * Rb * Rc + (Ag + 2) * Rc * Rc) * Rd) * Ri) * Ro), (((Ag + 1) * Rb * Rb + 2 * (Ag + 1) * Rb * Rc) * Rd * Rd * Ri + ((Ag * Ag + 3 * Ag + 2) * Rb * Rd * Rd + ((Ag * Ag + 2 * Ag + 1) * Rb * Rb + 2 * (Ag * Ag + 2 * Ag + 1) * Rb * Rc) * Rd) * Ri * Ri + ((2 * Rb + Rc) * Rd * Ri + Rd * Ri * Ri + (Rb * Rb + Rb * Rc) * Rd) * Ro * Ro - ((Rb * Rb + Rb * Rc) * Rd * Rd + ((Ag + 2) * Rd * Rd + 2 * ((Ag + 1) * Rb + (Ag + 1) * Rc) * Rd) * Ri * Ri + (((Ag + 3) * Rb + (Ag + 2) * Rc) * Rd * Rd + (2 * (Ag + 1) * Rb * Rb + 3 * (Ag + 1) * Rb * Rc) * Rd) * Ri) * Ro) / ((Rb * Rb * Rc + Rb * Rc * Rc) * Rd * Rd + ((Ag * Ag + 2 * Ag + 1) * Rb * Rb * Rc + (Ag * Ag + 2 * Ag + 1) * Rb * Rc * Rc + (Ag + 1) * Rb * Rd * Rd + ((Ag * Ag + 2 * Ag + 1) * Rb * Rb + (Ag * Ag + 3 * Ag + 2) * Rb * Rc) * Rd) * Ri * Ri + (Rb * Rb * Rc + Rb * Rc * Rc + (Rc + Rd) * Ri * Ri + (Rb * Rb + 2 * Rb * Rc + Rc * Rc) * Rd + (2 * Rb * Rc + Rc * Rc + 2 * (Rb + Rc) * Rd) * Ri) * Ro * Ro + (((Ag + 1) * Rb * Rb + (Ag + 2) * Rb * Rc) * Rd * Rd + 2 * ((Ag + 1) * Rb * Rb * Rc + (Ag + 1) * Rb * Rc * Rc) * Rd) * Ri - ((Rb * Rb + 2 * Rb * Rc + Rc * Rc) * Rd * Rd + (2 * (Ag + 1) * Rb * Rc + (Ag + 1) * Rc * Rc + (2 * (Ag + 1) * Rb + (Ag + 2) * Rc) * Rd + Rd * Rd) * Ri * Ri + 2 * (Rb * Rb * Rc + Rb * Rc * Rc) * Rd + (2 * (Ag + 1) * Rb * Rb * Rc + 2 * (Ag + 1) * Rb * Rc * Rc + 2 * (Rb + Rc) * Rd * Rd + (2 * (Ag + 1) * Rb * Rb + 3 * (Ag + 2) * Rb * Rc + (Ag + 2) * Rc * Rc) * Rd) * Ri) * Ro), -(((Ag + 1) * Rb * Rb + (Ag + 2) * Rb * Rc) * Rd * Rd * Ri + (Rb * Rb * Rc + Rb * Rc * Rc) * Rd * Rd - ((Ag * Ag + 2 * Ag + 1) * Rb * Rb * Rc + (Ag * Ag + 2 * Ag + 1) * Rb * Rc * Rc - (Ag + 1) * Rb * Rd * Rd) * Ri * Ri - (Rb * Rb * Rc + Rb * Rc * Rc + Rc * Ri * Ri + (2 * Rb * Rc + Rc * Rc) * Ri) * Ro * Ro - ((Rb * Rb + 2 * Rb * Rc + Rc * Rc) * Rd * Rd - (2 * (Ag + 1) * Rb * Rc + (Ag + 1) * Rc * Rc - Rd * Rd) * Ri * Ri - 2 * ((Ag + 1) * Rb * Rb * Rc + (Ag + 1) * Rb * Rc * Rc - (Rb + Rc) * Rd * Rd) * Ri) * Ro) / ((Rb * Rb * Rc + Rb * Rc * Rc) * Rd * Rd + ((Ag * Ag + 2 * Ag + 1) * Rb * Rb * Rc + (Ag * Ag + 2 * Ag + 1) * Rb * Rc * Rc + (Ag + 1) * Rb * Rd * Rd + ((Ag * Ag + 2 * Ag + 1) * Rb * Rb + (Ag * Ag + 3 * Ag + 2) * Rb * Rc) * Rd) * Ri * Ri + (Rb * Rb * Rc + Rb * Rc * Rc + (Rc + Rd) * Ri * Ri + (Rb * Rb + 2 * Rb * Rc + Rc * Rc) * Rd + (2 * Rb * Rc + Rc * Rc + 2 * (Rb + Rc) * Rd) * Ri) * Ro * Ro + (((Ag + 1) * Rb * Rb + (Ag + 2) * Rb * Rc) * Rd * Rd + 2 * ((Ag + 1) * Rb * Rb * Rc + (Ag + 1) * Rb * Rc * Rc) * Rd) * Ri - ((Rb * Rb + 2 * Rb * Rc + Rc * Rc) * Rd * Rd + (2 * (Ag + 1) * Rb * Rc + (Ag + 1) * Rc * Rc + (2 * (Ag + 1) * Rb + (Ag + 2) * Rc) * Rd + Rd * Rd) * Ri * Ri + 2 * (Rb * Rb * Rc + Rb * Rc * Rc) * Rd + (2 * (Ag + 1) * Rb * Rb * Rc + 2 * (Ag + 1) * Rb * Rc * Rc + 2 * (Rb + Rc) * Rd * Rd + (2 * (Ag + 1) * Rb * Rb + 3 * (Ag + 2) * Rb * Rc + (Ag + 2) * Rc * Rc) * Rd) * Ri) * Ro) ]])


        Ra = (Rb * Rc * Rd + ((Ag + 1) * Rb * Rc + (Ag + 1) * Rb * Rd) * Ri - (Rb * Rc + (Rb + Rc) * Rd + (Rc + Rd) * Ri) * Ro) / ((Rb + Rc) * Rd + ((Ag + 1) * Rb + (Ag + 1) * Rc + Rd) * Ri - (Rb + Rc + Ri) * Ro)
        return Ra

    def set_params(self, cutoff: float, q_val: float) -> None:
        Rval = 1 / (self.capVal * 2* np.pi * cutoff)

        # q_val = clamp(.01, self.capRatio * .5, q_val)
        # Rratio = self.capRatio * np.sqrt(self.capRatio * self.capRatio - 4 * q_val * q_val) / 2 * q_val
        R_ratio = 0.64174
        self.R1.set_resistance(Rval * R_ratio)
        self.R2.set_resistance(Rval / R_ratio)
        print(self.R1, self.R2)
        print(self.C1, self.C2)


if __name__ == '__main__':
    skf = SallenKeyFilter(44100, 1000, .5)
    skf.set_params(100, 2)
    root_dir = src_dir.parent
    out_path = root_dir / "data" / script_path.stem / "freqz.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    skf.plot_freqz(out_path)

