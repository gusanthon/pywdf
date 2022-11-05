## wdf.py
<code>wdf.py</code> is a lightweight Python library for modeling and simulating wave digital filter circuits.

Based on Jatin Chowdhury's C++ wdf library:  https://github.com/Chowdhury-DSP/chowdsp_wdf

Associated with master's thesis:

## Usage

Basic wdf elements and adaptors can be found in<code>wdf.py</code>, while adapted and unadapted R-type adaptors are in <code>rtype.py</code>.  <code>circuit.py</code> contains a generic circuit class from which any wave digital circuit built using this library can inherit basic functionalities, such as  <code>process_sample</code>, <code>get_impulse_response</code>, <code>plot_freqz</code>.

Below is an example RC low pass filter, initialized with its components and connections, and inheriting the Circuit functionality by specifying its list of elements, its voltage source, the root of its connection tree, and where to probe the output voltage. It is equivalent to the RC low pass filter example provided in Jatin Chowdhury's C++ library above. 

```python
from circuit import Circuit

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

        elements = [
            self.R1,
            self.C1,
            self.S1,
            self.I1,
            self.Vs
        ]

        super().__init__(elements, self.Vs, self.Vs, self.C1)
      
      
    def set_cutoff(self, new_cutoff: float):
        if self.cutoff != new_cutoff:
            self.cutoff = new_cutoff
            self.R = 1.0 / (2 * np.pi * self.C * self.cutoff)
            self.R1.set_resistance(self.R)

lpf = RCLowPass(44100, 1000)
lpf.set_cutoff(2000)
lpf.plot_freqz()
```

Also in <code>circuit.py</code> are several example circuits, including <code>DiodeClipper</code>, <code>TR_808_HatResonator</code>, <code>BaxandallEQ</code>, e.g:
```python
from circuit import DiodeClipper

dc = DiodeClipper(sample_rate=44100)
dc.set_input_gain(5)
dc.set_cutoff(5000)

dc.plot_freqz()

y = dc.process_wav('path/to/file.wav')

```


