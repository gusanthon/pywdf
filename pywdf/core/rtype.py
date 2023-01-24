import numpy as np
from .wdf import baseWDF, rootWDF
from typing import Callable


class RTypeAdaptor(baseWDF):
    def __init__(
        self, down_ports: list, impedance_calc: Callable, up_port_idx: int
    ) -> None:
        baseWDF.__init__(self)

        if up_port_idx is not None:
            self.n_ports = len(down_ports) + 1
            self.up_port_idx = up_port_idx

        else:
            self.n_ports = len(down_ports)

        self.down_ports = down_ports
        self.impedance_calc = impedance_calc
        self.S_matrix = np.zeros((self.n_ports, self.n_ports))
        self.a_vals = np.zeros(self.n_ports)
        self.b_vals = np.zeros(self.n_ports)

        for port in self.down_ports:
            port.connect_to_parent(self)

        self.calc_impedance()

    def reset(self) -> None:
        self.a_vals = np.zeros(self.n_ports)
        self.b_vals = np.zeros(self.n_ports)

    def accept_incident_wave(self, a: float) -> None:
        self.a = a
        self.a_vals[self.up_port_idx] = self.a
        self.r_type_scatter()
        for i in range(len(self.down_ports)):
            idx = self.get_port_idx(i)
            self.down_ports[i].accept_incident_wave(self.b_vals[idx])

    def propagate_reflected_wave(self) -> float:
        for i in range(len(self.down_ports)):
            idx = self.get_port_idx(i)
            self.a_vals[idx] = self.down_ports[i].propagate_reflected_wave()
        self.b = self.b_vals[self.up_port_idx]
        return self.b

    def get_port_idx(self, x: int) -> int:
        return x if x < self.up_port_idx else x + 1

    def r_type_scatter(self) -> None:
        for i in range(self.n_ports):
            self.b_vals[i] = 0
            for j in range(self.n_ports):
                self.b_vals[i] += self.S_matrix[i][j] * self.a_vals[j]

    def calc_impedance(self) -> None:
        self.Rp = self.impedance_calc(self)
        self.G = 1.0 / self.Rp

    def get_port_impedances(self) -> list:
        return [port.Rp for port in self.down_ports]

    def set_S_matrix(self, matrix: np.array) -> None:
        for i in range(self.n_ports):
            for j in range(self.n_ports):
                self.S_matrix[i][j] = matrix[i][j]


class RootRTypeAdaptor(RTypeAdaptor, rootWDF):
    def __init__(self, down_ports: list, impedance_calc: Callable) -> None:
        super().__init__(down_ports, impedance_calc, None)

    def compute(self) -> None:
        self.r_type_scatter()
        for i in range(len(self.down_ports)):
            self.down_ports[i].accept_incident_wave(self.b_vals[i])
            self.a_vals[i] = self.down_ports[i].propagate_reflected_wave()

    def calc_impedance(self) -> None:
        self.impedance_calc(self)
