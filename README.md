## pywdf
<code>pywdf</code> is a Python framework for modeling and simulating wave digital filter circuits.

Based on Jatin Chowdhury's C++ [WDF library](https://github.com/Chowdhury-DSP/chowdsp_wdf)  

Developed from work done in master thesis [Evaluating the Nonlinearities of A Diode Clipper Circuit Based on Wave Digital Filters](https://zenodo.org/record/7116075) 

## Dependencies
- [Numpy](numpy.org)
- [Scipy](scipy.org)
- [Matplotlib](https://matplotlib.org/)


## Installation
<code>pip install git+https://github.com/gusanthon/pywdf</code>

## Usage

The <code>core</code> directory contains the main source code for the framework. Basic wave digital elements and series and parallel adaptors can be found in <code>wdf.py</code>, while <code>rtype.py</code> contains adapted and unadapted R-type adaptors.  <code>circuit.py</code> contains a generic circuit class from which any wave digital circuit built using this library can inherit basic functionalities, such as  <code>process_sample</code>, <code>get_impulse_response</code>, <code>plot_freqz</code>.

Below is an example RC low pass filter, initialized with its components and connections, and inheriting the Circuit class functionality by specifying its voltage source, the root of its connection tree, and which wave digital element to probe for the output voltage. It is equivalent to the RC low pass filter [example](https://github.com/Chowdhury-DSP/chowdsp_wdf#basic-usage) provided in Jatin Chowdhury's C++ library. 

```python
from pywdf.core.wdf import *
from pywdf.core.circuit import *


class RCLowPass(Circuit):
    def __init__(self, sample_rate: int, cutoff: float) -> None:
    
        self.fs = sample_rate
        self.cutoff = cutoff
        
        self.C = 1e-6
        self.R = 1.0 / (2 * np.pi * self.C * self.cutoff)

        self.R1 = Resistor(self.R)
        self.C1 = Capacitor(self.C, self.fs)

        self.S1 = SeriesAdaptor(self.R1, self.C1)
        self.I1 = PolarityInverter(self.S1)
        self.Vs = IdealVoltageSource(self.I1)

        super().__init__(self.Vs, self.Vs, self.C1)
      
    def set_cutoff(self, new_cutoff: float):
        if self.cutoff != new_cutoff:
            self.cutoff = new_cutoff
            self.R = 1.0 / (2 * np.pi * self.C * self.cutoff)
            self.R1.set_resistance(self.R)

lpf = RCLowPass(44100, 1000)
lpf.set_cutoff(2000)
lpf.plot_freqz()
```

See examples directory for several example circuits, including <code>DiodeClipper</code>, <code>RCA_MK2_SEF</code>, <code>TR_808_HatResonator</code> and others. Usage:

```python
import pywdf

# sweep positions of RCA mk2 SEF low pass filter knob and plot frequency responses
mk2_sef = pywdf.RCA_MK2_SEF(44100, 0, 3000)
positions = range(1,11)
mk2_sef.plot_freqz_list(positions, mk2_sef.set_lowpass_knob_position, param_label = 'lpf knob pos')

# sweep resonance values in tr 808 hat resonator and plot frequency responses
hr = pywdf.TR_808_HatResonator(44100, 1000, .5)
resonances = [.1,.2,.3,.4,.5,.6,.7,.8,.9]
hr.plot_freqz_list(resonances, hr.set_resonance, param_label = 'resonance val')

# analyze transient response of Diode Clipper to AC signal
dc = pywdf.DiodeClipper(44100, cutoff= 1000, input_gain_db = 5)
dc.AC_transient_analysis()
```