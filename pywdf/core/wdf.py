from __future__ import annotations
import numpy as np
from enum import Enum
import math 

class Transform(Enum):
    BILINEAR = 1
    WARPED_BILINEAR = 2
    BACKWARD_EULER = 3
    ALPHA = 4
    MOBIUS = 5


class baseWDF:
    '''
    The base one port object from which all wave digital elements will inherit.
    '''
    def __init__(self) -> None:
        self.a = 0.0
        self.b = 0.0
        # Added default values for impedance and admittance
        self.Rp = 1.0e-9         # impedance 
        self.G = 1.0 / self.Rp   # admittance 
        self.parent = None

    def connect_to_parent(self, parent: baseWDF) -> None:
        self.parent = parent

    def accept_incident_wave(self, a: float) -> None:
        self.a = a

    def impedance_change(self) -> None:
        self.calc_impedance()
        if self.parent != None:
            self.parent.impedance_change()

    def wave_to_voltage(self) -> float:
        return (self.a + self.b) * 0.5

    # Added base class method to convert wave to current according to WDF parametric definition
    def wave_to_current(self) -> float:
        return (self.a - self.b) * 0.5 * self.G   

    def reset(self) -> None:
        self.a, self.b = 0, 0

    def calc_impedance(self) -> None:
        pass

    def propagate_reflected_wave(self) -> float:
        pass

    def __str__(self) -> str:
        return f"{self.__class__.__name__}, ({self.__dict__})"


class rootWDF(baseWDF):
    '''
    The rootWDF class is a subclass of baseWDF that represents the root element in a WDF circuit.
    It inherits from baseWDF but overwrites the connection of an element to a parent 
    and throws an error should this be attempted.
    '''
    def __init__(self, next: baseWDF) -> None:
        baseWDF.__init__(self)
        self.next = next
        next.connect_to_parent(self)

    def connect_to_parent(self, p: baseWDF) -> None:
        raise Exception("Root elements cannot be connected to a parent!")


####################################################################################


# open circuit, close circuit and switch (they can be seen as a variable resistors)
class ShortCircuit(baseWDF):
    def __init__(self):
        baseWDF.__init__(self)
        self.calc_impedance()

    def calc_impedance(self) -> None:
        # self.Rp = 0
        # self.G = 1e16  
        self.Rp = 1e-16  # short circuit has zero resistance, here we set it to a very small value
        self.G = 1.0 / self.Rp  # set the conductance to a very large value

    def accept_incident_wave(self, a: float) -> None:
        self.a = a

    def propagate_reflected_wave(self) -> float:
        self.b = -self.a
        return self.b


####################################################################################


class OpenCircuit(baseWDF):
    def __init__(self):
        baseWDF.__init__(self)
        self.calc_impedance()

    def calc_impedance(self) -> None:
        # self.Rp = 0
        # self.G = 1e16  
        self.Rp = 1e16  # open circuit has infinite resistance, here we set it to a very large value
        self.G = 1.0 / self.Rp  # set the conductance to a very small value

    def accept_incident_wave(self, a: float) -> None:
        self.a = a

    def propagate_reflected_wave(self) -> float:
        self.b = self.a
        return self.b


####################################################################################


class Resistor(baseWDF):
    def __init__(self, R: float = 1e-9) -> None:
        baseWDF.__init__(self)
        self.Rp = R
        self.calc_impedance()

    def set_resistance(self, new_R: float) -> None:
        if self.Rp != new_R:
            self.Rp = new_R
            self.impedance_change()

    def calc_impedance(self) -> None:
        self.G = 1.0 / self.Rp

    def propagate_reflected_wave(self) -> float:
        self.b = 0
        return self.b


####################################################################################


class Capacitor(baseWDF):
    def __init__(
            self, 
            C: float, 
            fs: int, 
            tolerance: float = 0,
            alpha: float = 1.0      # 0.0 – Backward Euler; 1.0 – Bilinear Transform 
        ) -> None:

        baseWDF.__init__(self)
        self.fs = fs
        self.tolerance = tolerance
        rand_samp = np.random.normal(loc=0, scale=C * tolerance / 2)
        self.C = C + rand_samp
        self.z = 0
        self.setAlpha(alpha)
        self.calc_impedance()


    def setAlpha(self, alpha: float) -> None:
        self.alpha = alpha
        self.b_coef = (1.0 - alpha) / 2.0
        self.a_coef = (1.0 + alpha) / 2.0
        self.impedance_change()


    def prepare(self, new_fs: int) -> None:
        self.fs = new_fs
        self.impedance_change()
        self.reset()

    def set_capacitance(self, new_C: float) -> None:
        if self.C != new_C:
            self.C = new_C
            self.impedance_change()

    # def calc_impedance(self) -> None:
    #     '''
    #     Local discretisation of capacitor using the warped bilinear transform method.

    #                 1
    #     Z_C = --------------
    #             2 * C * f_s
        
    #     where f_s is the sampling frequency and C the capacitance.
    #     '''
        
    #     self.G = 1.0 / self.Rp


    def calc_impedance(
            self, 
        ) -> None:

        self.Rp = 1.0 / ( ( 1.0 + self.alpha ) * self.C * self.fs)
        self.G = 1.0 / self.Rp

    def accept_incident_wave(self, a: float) -> None:
        self.a = a
        self.z = self.a

    def propagate_reflected_wave(self) -> float:
        # self.b = self.z
        self.b = self.b_coef * self.b + self.a_coef * self.z
        return self.b

    def reset(self) -> None:
        super().reset()
        self.z = 0


####################################################################################


class Inductor(baseWDF):
    
    def __init__(
        self, 
        L: float, 
        fs: int,
        alpha: float = 1.0
    ) -> None:

        baseWDF.__init__(self)
        self.fs = fs
        self.L = L
        self.z = 0.0
        self.alpha = alpha
        self.b_coef = (1.0 - alpha) / 2.0
        self.a_coef = (1.0 + alpha) / 2.0        
        self.calc_impedance()

    def prepare(self, new_fs: int) -> None:
        self.fs = new_fs
        self.impedance_change()
        self.reset()

    def setAlpha(self, alpha: float) -> None:
        self.alpha = alpha
        self.b_coef = (1.0 - alpha) / 2.0
        self.a_coef = (1.0 + alpha) / 2.0
        self.impedance_change()

    def set_inductance(self, new_L: float) -> None:
        if self.L != new_L:
            self.L = new_L
            self.impedance_change()

    def calc_impedance(
            self
        ) -> None:

        self.Rp = (1.0 + self.alpha ) * self.L * self.fs

        self.G = 1.0 / self.Rp
    
    def accept_incident_wave(self, a: float) -> None:
        self.a = a
        self.z = self.a

    def propagate_reflected_wave(self) -> float:
        # self.b = -self.z
        self.b = self.b_coef * self.b - self.a_coef * self.z
        return self.b

    def reset(self) -> None:
        super().reset()
        self.z = 0


####################################################################################


class ParallelAdaptor(baseWDF):
    def __init__(self, 
                 p1: baseWDF, 
                 p2: baseWDF
        ) -> None:
        baseWDF.__init__(self)
        self.p1 = p1
        self.p2 = p2
        self.b_temp = 0
        self.b_diff = 0
        self.p1_reflect = 1
        p1.connect_to_parent(self)
        p2.connect_to_parent(self)
        self.calc_impedance()

    def calc_impedance(self) -> None:
        self.G = self.p1.G + self.p2.G
        self.Rp = 1.0 / self.G
        self.p1_reflect = self.p1.G / self.G

    def accept_incident_wave(self, a: float) -> None:
        b2 = a + self.b_temp
        self.p1.accept_incident_wave(self.b_diff + b2)
        self.p2.accept_incident_wave(b2)
        self.a = a

    def propagate_reflected_wave(self) -> float:
        self.b_diff = (
            self.p2.propagate_reflected_wave() - self.p1.propagate_reflected_wave()
        )
        self.b_temp = -self.p1_reflect * self.b_diff
        self.b = self.p2.b + self.b_temp
        return self.b


####################################################################################


class SeriesAdaptor(baseWDF):
    def __init__(self, p1: baseWDF, p2: baseWDF) -> None:
        baseWDF.__init__(self)
        self.p1 = p1
        self.p2 = p2
        self.p1_reflect = 1
        p1.connect_to_parent(self)
        p2.connect_to_parent(self)
        self.calc_impedance()

    def calc_impedance(self) -> None:
        self.Rp = self.p1.Rp + self.p2.Rp
        self.G = 1.0 / self.Rp
        self.p1_reflect = self.p1.Rp / self.Rp

    def accept_incident_wave(self, a: float) -> None:
        b1 = self.p1.b - self.p1_reflect * (a + self.p1.b + self.p2.b)
        self.p1.accept_incident_wave(b1)
        self.p2.accept_incident_wave(0 - (a + b1))
        self.a = a

    def propagate_reflected_wave(self) -> float:
        self.b = -(
            self.p1.propagate_reflected_wave() + self.p2.propagate_reflected_wave()
        )
        return self.b


####################################################################################


class Switch(baseWDF):
    def __init__(self, next: baseWDF):
        rootWDF.__init__(self, next)
        next.connect_to_parent(self)
        self.closed = True

    def accept_incident_wave(self, a: float) -> None:
        self.a = a

    def propagate_reflected_wave(self) -> float:
        if self.closed:  # Closed
            self.b = -self.a
        else:  # Open
            self.b = self.a
        return self.b

    def set_closed(self, closed: bool) -> None:
        self.closed = closed


####################################################################################


class PolarityInverter(baseWDF):
    def __init__(self, p1: baseWDF) -> None:
        baseWDF.__init__(self)
        p1.connect_to_parent(self)
        self.p1 = p1
        self.calc_impedance()

    def calc_impedance(self) -> None:
        self.Rp = self.p1.Rp
        self.G = 1.0 / self.Rp

    def accept_incident_wave(self, a: float) -> None:
        self.a = a
        self.p1.accept_incident_wave(-a)

    def propagate_reflected_wave(self) -> float:
        self.b = 0 - self.p1.propagate_reflected_wave()
        return self.b


####################################################################################


class SeriesVoltage(baseWDF):
    
    def __init__(self, p1: baseWDF) -> None:
        baseWDF.__init__(self)
        p1.connect_to_parent(self)
        self.p1 = p1
        self.calc_impedance()

    def calc_impedance(self) -> None:
        self.Rp = self.p1.Rp
        self.G = 1.0 / self.Rp

    def set_voltage(self, new_V: float) -> None:
        self.Vs = new_V

    def accept_incident_wave(self, a: float) -> None:
        self.a = a
        self.p1.accept_incident_wave( -a + -self.Vs )  # Incident wave is the negative of the voltage source

    def propagate_reflected_wave(self) -> float:
        self.b = 0 - self.p1.propagate_reflected_wave() + -self.Vs
        return self.b



####################################################################################


class IdealVoltageSource(rootWDF):
    def __init__(self, next: baseWDF) -> None:
        '''
        IdealVoltageSource is a subclass of rootWDF and not of baseWDF.
        This is because there is delay free loop in the reflected wave and therefore
        we can only use an ideal voltage source at the root of a connection tree, not as a leaf node.
        '''
        rootWDF.__init__(self, next)
        self.Vs = 0
        self.calc_impedance()

    def calc_impedance(self) -> None:
        self.Rp = self.next.Rp
        self.G = 1.0 / self.Rp

    def set_voltage(self, new_V: float) -> None:
        self.Vs = new_V

    def accept_incident_wave(self, a: float) -> None:
        self.a = a

    def propagate_reflected_wave(self) -> float:
        self.b = -self.a + 2.0 * self.Vs
        return self.b


####################################################################################


class IdealCurrentSource(rootWDF):

    def __init__(self, next: baseWDF) -> None: 
        '''
        IdealCurrentSource is a subclass of rootWDF and not of baseWDF.
        This is because there is delay free loop in the reflected wave  
        therefore we can only use an ideal current source at the root of a connection tree, not as a leaf node.
        '''
        rootWDF.__init__(self, next)
        self.next = next
        self.Is = 0
        self.twoR = 0
        self.twoR_Is = 0
        self.calc_impedance()

    def calc_impedance(self) -> None:
        self.Rp = self.next.Rp
        self.G = 1.0 / self.Rp
        self.twoR = 2.0 * self.next.Rp
        self.twoR_Is = self.twoR * self.Is

    def set_current(self, new_I: float) -> None:
        self.Is = new_I
        self.twoR_Is = self.twoR * self.Is
       

    def accept_incident_wave(self, a: float) -> None:
        self.a = a

    def propagate_reflected_wave(self) -> float:
        self.b = self.a + self.twoR_Is 
        return self.b


####################################################################################


class ResistiveVoltageSource(baseWDF):
    def __init__(self, Rval: float = None) -> None: 
    # def __init__(self, next: baseWDF = None, Rval: float = None) -> None:
        baseWDF.__init__(self)
        self.Rval = Rval if Rval else 1e-9
        self.Vs = 0
        self.calc_impedance()

    def set_resistance(self, new_R: float) -> None:
        if self.Rval != new_R:
            self.Rval = new_R
            self.impedance_change()

    def calc_impedance(self) -> None:
        self.Rp = self.Rval
        self.G = 1.0 / self.Rp

    def set_voltage(self, new_V: float) -> None:
        self.Vs = new_V

    def propagate_reflected_wave(self) -> None:
        self.b = self.Vs
        return self.b


####################################################################################



class ChuaDiode(rootWDF):
    """
    Computes the Chua diode nonlinearity, a piecewise linear active resistance.
    This class implements a Chua diode model for Wave Digital Filter (WDF) circuits.
    The Chua diode is characterized by a piecewise linear current-voltage relationship.
    """
    
    def __init__(
            self, 
            next: baseWDF, 
            g1: float, 
            g2: float, 
            v0: float, 
            r1: float
        ) -> None:
        """
        Initialize the Chua diode.
        
        Args:
            next_element: The next WDF element in the circuit
            g1: Slope of i(v) parameter of the chua diode
            g2: Slope of i(v) parameter of the chua diode
            v0: Voltage parameter of the chua diode
            r1: Resistance value in Ohms
        """
        rootWDF.__init__(self, next)
        next.connect_to_parent(self)
        self.set_Chua_parameters(g1, g2, r1, v0)
        
    
    def set_Chua_parameters(
            self, 
            g1: float, 
            g2: float, 
            r1: float, 
            v0: float,
        ) -> None:
        """
        Sets diode specific parameters.
        
        Args:
            g1: First conductance parameter
            g2: Second conductance parameter  
            r1: Resistance parameter
        """
        self.G1 = (1.0 - g1 * r1) / (1.0 + g1 * r1)
        self.G2 = (1.0 - g2 * r1) / (1.0 + g2 * r1)
        self.a_0 = v0 * (1.0 + g2 * r1)


    def calc_impedance(self):
        """Calculate impedance - implementation depends on specific circuit context"""
        # Implementation would depend on the specific WDF framework
        # This is a placeholder as the original C++ implementation is empty
        pass
    
     
    def incident(self, a: float):
        """
        Set the incident wave.
        
        Args:
            a: The incident wave value
        """
        self.a = a
    
       
    def propagate_reflected_wave(self) -> float:
        """
        Compute the reflected wave using the Chua diode characteristic.
        This implements the piecewise linear characteristic of the Chua diode
        using the absolute value function to create the nonlinear behavior.
        
        Returns:
            The reflected wave value
        """
        # TODO: g1 and g2 do not need to be calculated for every reflected calculation, just a_0

        
        self.b = ( self.G1 * self.a + 0.5 * (self.G2 - self.G1) * (abs( self.a + self.a_0 ) - abs( self.a - self.a_0 )))
        
        return self.b
        


####################################################################################


class Diode(rootWDF):
    def __init__(
        self, 
        next: baseWDF, 
        Is: float, 
        Vt: float = 25.85e-3, 
        n_diodes: float = 1
    ) -> None:
        
        rootWDF.__init__(self, next)
        next.connect_to_parent(self)
        self.set_diode_params(Is, Vt, n_diodes)


    def set_diode_params(self, Is: float, Vt: float, n_diodes: float) -> None:
        self.Is = Is
        self.n_diodes = n_diodes
        self.Vt = Vt * n_diodes
        self.one_over_Vt = 1.0 / self.Vt
        self.calc_impedance()

    def set_n_diodes(self, n_diodes: float, Vt: float = 25.85e-3) -> None:
        if self.n_diodes != n_diodes:
            self.n_diodes = n_diodes
            self.Vt = Vt * n_diodes
            self.one_over_Vt = 1.0 / self.Vt
            self.calc_impedance()

    def calc_impedance(self) -> None:
        self.two_R_Is = 2.0 * self.next.Rp * self.Is
        self.R_Is_over_Vt = self.next.Rp * self.Is * self.one_over_Vt
        self.logR_Is_over_Vt = np.log(self.R_Is_over_Vt)

    def propagate_reflected_wave(self) -> float:
        self.b = (
            self.a
            + self.two_R_Is
            - (2.0 * self.Vt)
            * self.omega4(
                self.logR_Is_over_Vt + self.a * self.one_over_Vt + self.R_Is_over_Vt
            )
        )
        return self.b

    def omega4(self, x: float) -> float:
        """
        4th order approximation of Wright Omega function
        y = 3rd order approx, is used in calculation of 4th
        """
        x1 = -3.341459552768620
        x2 = 8.0
        a = -1.314293149877800e-3
        b = 4.775931364975583e-2
        c = 3.631952663804445e-1
        d = 6.313183464296682e-1
        if x < x1:
            y = 0
        elif x < x2:
            y = d + x * (c + x * (b + x * a))
        else:
            y = x - np.log(x)
        return y - (y - np.exp(x - y)) / (y + 1)


class DiodePair(Diode):
    def __init__(
        self, 
        next: baseWDF, 
        Is: float, 
        Vt: float = 25.85e-3, 
        n_diodes: float = 2
    ) -> None:
        Diode.__init__(self, next, Is, Vt, n_diodes)

    def propagate_reflected_wave(self) -> float:
        lam = np.sign(self.a)
        lam_a_over_Vt = lam * self.a * self.one_over_Vt
        self.b = self.a - (2 * self.Vt) * lam * (
            self.omega4(self.logR_Is_over_Vt + lam_a_over_Vt)
            - self.omega4(self.logR_Is_over_Vt - lam_a_over_Vt)
        )
        return self.b
