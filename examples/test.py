import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from matplotlib.pyplot import cm

from circuit import *

from baxandalleq import *

fs = 44100

# hr = TR_808_HatResonator(fs,2000,.04)
# hr.plot_freqz( )

# ubeq = UnadaptedBaxandallEQ(fs, .5,.5)
# ubeq.plot_freqz()

colors = ['grey','olive','purple','green','pink','brown','cyan','red','orange','blue']

def freqz(x, fs):
    w, h = scipy.signal.freqz(x, 1, 4096)
    H = 20 * np.log10(np.abs(h))
    f = w / (2 * np.pi) * fs
    angles = np.unwrap(np.angle(h))
    return f, H, angles

def plot_magnitude_response(f, H, label="magnitude", c="tab:blue", title=""):
    ax = plt.subplot(111)
    ax.plot(f, H)
    ax.semilogx(f, H, label=label, color=c)
    plt.ylabel("Amplitude [dB]")
    plt.xlabel("Frequency [hz]")
    plt.title(title + "Magnitude response")

def plot_phase_response(
    f, angles, mult_locater=(np.pi / 2), denom=2, label="phase", c="tab:blue", title=""
):
    ax = plt.subplot(111)
    plt.plot(f, angles)
    ax.semilogx(f, angles, label=label, color=c)
    plt.ylabel("Angle [radians]")
    plt.xlabel("Frequency [hz]")
    plt.title(title + "Phase response")
    # ax.yaxis.set_major_locator(plt.MultipleLocator(mult_locater))
    # ax.yaxis.set_major_formatter(
    #     plt.FuncFormatter(multiple_formatter(denominator=denom))
    # )



beq = BaxandallEQ(44100,.5,.5)
ubeq = UnadaptedBaxandallEQ(44100,.5,.5)
# hr = TR_808_HatResonator(44100,.5,.5)

# def plot_response(*args):
#     i=0

#     for t in treble:
#         for b in bass:
#             # ubeq = UnadaptedBaxandallEQ(44100,b,t)
#             # ubeq = TR_808_HatResonator(44100,b,t)
#             x = ubeq.get_impulse_response()
#             f,H,_ = freqz(x,beq.fs)
#             plot_magnitude_response(f,H,c=colors[i%len(colors)])

#             # x = ubeq.get_impulse_response()
#             # f,H,_ = freqz(x,beq.fs)
#             # plot_magnitude_response(f,H,c=colors[i%len(colors)])
#             i+=1

def plot_response(circuit,param1, param2,plot='mag'):
    i=0
    for a in param1:
        for b in param2:
            c = circuit(44100,a,b)
            x = c.get_impulse_response()
            f,H,angles = freqz(x,beq.fs)
            if plot == 'mag':
                plot_magnitude_response(f,H,c=colors[i%len(colors)])
            elif plot == 'phase':
                plot_phase_response(f,angles,c=colors[i%len(colors)])
            i+=1

# plot_response(UnadaptedBaxandallEQ,[.5,.9],[0.01, 0.25, 0.5, 0.75, 0.99],plot='phase')
plot_response(BaxandallEQ,[0.01, 0.25, 0.5, 0.75, 0.99],[0,.5,1])
# plot_response(TR_808_HatResonator,[1000,2000,3000,4000,5000,6000],[.01,.5,.1,])

# plot_response(PassiveLPF,[1000,2000,3000,4000,5000,6000])
# plt.grid()
# plt.show()

class Parameter:
    def __init__(self,value,min_val,max_val,setter,unit) -> None:
        self.value = value
        self.setter = setter
        self.min_val = min_val
        self.max_val = max_val
        self.unit = unit

# cutoff = Parameter(1000,20,20e3,plot_response,'hz')



# parameters = {
#     'cutoff' : {'value' : self.cutoff, 'setter' : self.set_cutoff}
# }

color = iter(cm.rainbow(np.linspace(0, 1, 10)))

i=0
# # for a in [0.01, 0.25, 0.5, 0.75, 0.99]:
# for a in [20,40,1000,2000,4000,8000,16000,20000,]:
#     # ubeq = UnadaptedBaxandallEQ(44100,.54,a)
#     p = PassiveLPF(44100,a)
#     x = p.get_impulse_response()
#     f,H,angles = freqz(x,ubeq.fs)
#     plot_magnitude_response(f,H,c=next(color))
#     # plot_phase_response(f,angles,c=next(color))
#     i+=1


# beq.plot_freqz()
# beq.set_bass(0)
# beq.set_treble(0)
# beq.plot_freqz()
# beq.set_bass(1)
# beq.set_treble(1)
# beq.plot_freqz()


#============================================= CURRENT DIVIDER TEST

# cd = CurrentDivider(44100,10000,10000)

# cd.plot_freqz()

class VoltageDivider(Circuit):
    def __init__(self, fs: int, R1_val: float, R2_val: float) -> None:

        self.fs = fs

        self.R1 = Resistor(R1_val)
        self.R2 = Resistor(R2_val)

        self.S1 = SeriesAdaptor(self.R1, self.R2)
        self.I1 = PolarityInverter(self.S1)
        self.Vs = IdealVoltageSource(self.I1)

        elements = [
            self.R1,
            self.R2,
            self.S1,
            self.I1,
            self.Vs,
        ]

        super().__init__(elements, self.Vs, self.Vs, self.R1)

vd = VoltageDivider(44100,1e-6,1e10)
# vd.plot_freqz()


#============================================= RCA FILTER TESTING ===================



# class RCAFilter(Circuit):
#     def __init__(self, sample_rate, lowpass_cutoff: float = 1000, highpass_cutoff: float = 200) -> None:
        
#         # L = 45.08e-3
#         # C = .272e-6
#         self.fs = sample_rate
#         L = 21.77e-3
#         C = .15e-6

#         k = 538.8

#         C_LP =  1 / k / (2 * np.pi * lowpass_cutoff)                    ### C_HP = (1 / kΩ) / (2π * cutoff)
#         L_LP =  self.fs / lowpass_cutoff * 2 * np.pi                       ### L_LP = 2 * L_HP  


#         C_HP = 2 * (1 / k / (2 * np.pi * highpass_cutoff))       ### C_LP = 2 * C_HP
#         L_HP = self.fs / lowpass_cutoff * 2 * np.pi                        ### 

#         self.C = C
#         self.L = L

#         Z_input = 560
#         Z_output = 560
        
#         self.fs = sample_rate

#         self.Rt = Resistor(Z_output)
#         self.L_LP4 = Inductor(L, self.fs)

#         self.S8 = SeriesAdaptor(self.L_LP4, self.Rt)
#         self.C_LP = Capacitor(C, self.fs)

#         self.S7 = SeriesAdaptor(self.C_LP, self.S8)
#         self.L_LP3 = Inductor(L, self.fs)

#         self.S6 = SeriesAdaptor(self.L_LP3, self.S7)
#         self.C_HP4 = Capacitor(C, self.fs)

#         self.S4 = SeriesAdaptor(self.C_HP4, self.S6)
#         self.L_HP2 = Inductor(L, self.fs)
        
#         self.P2 = ParallelAdaptor(self.L_HP2, self.S4)
#         self.C_HP3 = Capacitor(C, self.fs)

#         self.S3 = SeriesAdaptor(self.C_HP3, self.P2)
#         self.C_HP2 = Capacitor(C, self.fs)

#         self.S2 = SeriesAdaptor(self.C_HP2, self.S3)
#         self.L_LP1 = Inductor(L, self.fs)

#         self.P1 = ParallelAdaptor(self.L_LP1, self.S2)
#         self.C_HP1 = Capacitor(C, self.fs)

#         self.S1 = SeriesAdaptor(self.C_HP1, self.P1)
#         self.R0 = Resistor(Z_input)

#         self.S0 = SeriesAdaptor(self.S1, self.R0)
#         self.Vin = IdealVoltageSource(self.S1)

#         elements = [
#             self.Rt,
#             self.L_LP4,
#             self.S8,
#             self.C_LP, 
#             self.S7,
#             self.L_LP3,
#             self.S6,
#             self.C_HP4,
#             self.S4,
#             self.L_HP2,
#             self.P2,
#             self.C_HP3,
#             self.S3,
#             self.C_HP2,
#             self.S2,
#             self.L_LP1,
#             self.P1,
#             self.C_HP1,
#             self.S1,
#             self.Vin
#         ]

#         super().__init__(elements, self.Vin, self.Vin, self.Rt)

#     def __set_components(self):
#         self.C_HP1.set_capacitance()
#         self.C_HP2.set_capacitance()
#         self.C_HP3.set_capacitance()
#         self.C_HP4.set_capacitance()

#         # self.l
a = 'SEF_RCA_Filter'

class RCA_MK2_Filter(Circuit):
    def __init__(self, sample_rate, lowpass_cutoff: float = 1000, highpass_cutoff: float = 200) -> None:
        
        self.fs = sample_rate
        self.lowpass_cutoff = lowpass_cutoff
        self.highpass_cutoff = highpass_cutoff

        self.k = 538.8

        # self.C_HP = self.__get_C_HP(highpass_cutoff)
        # self.L_HP = 44324 * highpass_cutoff **-1

        # self.C_LP = self.__get_C_LP(lowpass_cutoff)                  
        # self.L_LP =  511e-3

        self.C_HP = 1.6e-6
        self.L_HP = 255.6e-3

        self.C_LP = .15e-6             
        self.L_LP =  22.38e-3


        Z_input = 560
        Z_output = 560
        
        self.fs = sample_rate

        self.Rt = Resistor(Z_output)
        self.L_LP5 = Inductor(self.L_LP, self.fs)

        self.S8 = SeriesAdaptor(self.L_LP5, self.Rt)
        self.C_LP2 = Capacitor(self.C_LP, self.fs)

        self.P4 = ParallelAdaptor(self.C_LP2, self.S8)
        self.L_LP4 = Inductor(self.L_LP, self.fs)

        self.S7 = SeriesAdaptor(self.L_LP4, self.P4)
        self.L_LP3 = Inductor(self.L_LP, self.fs)

        self.S6 = SeriesAdaptor(self.L_LP3, self.S7)
        self.C_LP1 = Capacitor(self.C_LP, self.fs)

        self.P3 = ParallelAdaptor(self.C_LP1, self.S6)
        self.L_LP2 = Inductor(self.L_LP, self.fs)

        self.S5 = SeriesAdaptor(self.L_LP2, self.P3)
        self.C_HP4 = Capacitor(self.C_HP, self.fs)

        self.S4 = SeriesAdaptor(self.C_HP4, self.S5)
        self.L_HP1 = Inductor(self.L_HP, self.fs)
        
        self.P2 = ParallelAdaptor(self.L_HP1, self.S4)
        self.C_HP3 = Capacitor(self.C_HP, self.fs)

        self.S3 = SeriesAdaptor(self.C_HP3, self.P2)
        self.C_HP2 = Capacitor(self.C_HP, self.fs)

        self.S2 = SeriesAdaptor(self.C_HP2, self.S3)
        self.L_LP1 = Inductor(self.L_LP, self.fs)

        self.P1 = ParallelAdaptor(self.L_LP1, self.S2)
        self.C_HP1 = Capacitor(self.C_HP, self.fs)

        self.S1 = SeriesAdaptor(self.C_HP1, self.P1)
        self.R0 = Resistor(Z_input)

        self.S0 = SeriesAdaptor(self.S1, self.R0)
        self.Vin = IdealVoltageSource(self.S1)

        elements = [
            self.Rt,
            self.L_LP5,
            self.S8,
            self.C_LP2,
            self.P4,
            self.L_LP4,
            self.S7,
            self.L_LP3,
            self.S6,
            self.C_LP1,
            self.P3,
            self.L_LP2,
            self.S5,
            self.C_HP4,
            self.S4,
            self.L_HP1,
            self.P2,
            self.C_HP3,
            self.S3,
            self.C_HP2,
            self.S2,
            self.L_LP1,
            self.P1,
            self.C_HP1,
            self.S1,
            self.R0, 
            self.S0,
            self.Vin
        ]

        super().__init__(elements, self.Vin, self.Vin, self.Rt)

    def __get_C_LP(self, cutoff):
        return 2 * self.__get_C_HP(cutoff)

    def __get_C_HP(self, cutoff):
        return (1/ self.k) / (2 * np.pi * cutoff)

    def __set_HP_components(self):
        self.C_HP1.set_capacitance(self.C_HP)
        self.C_HP2.set_capacitance(self.C_HP)
        self.C_HP3.set_capacitance(self.C_HP)
        self.C_HP4.set_capacitance(self.C_HP)

        self.L_HP1.set_inductance(self.L_HP)

    def __set_LP_components(self):
        self.C_LP1.set_capacitance(self.C_LP)
        self.C_LP2.set_capacitance(self.C_LP)

        self.L_LP1.set_inductance(self.L_LP1)
        self.L_LP2.set_inductance(self.L_LP)
        self.L_LP3.set_inductance(self.L_LP)
        self.L_LP4.set_inductance(self.L_LP)
        self.L_LP5.set_inductance(self.L_LP)

    def set_highpass_cutoff(self, new_cutoff: float ):
        if self.highpass_cutoff != new_cutoff:
            self.C_HP = self.__get_C_HP(new_cutoff)
            self.L_HP = self.fs / self.highpass_cutoff * 2 * np.pi
            self.__set_HP_components()

    def set_lowpass_cutoff(self, new_cutoff: float):
        if self.lowpass_cutoff != new_cutoff:
            self.C_LP = self.__get_C_LP(new_cutoff)
            self.L_LP = ...
            self.__set_LP_components()

# r = Resistor()
# if r.__class__.__name__ == "Resistor":
#     print('f™™£ƒ£®ƒˆœ∆ø®ƒ£ˆœ®∆ƒsdfadkslfj')

rca= RCA_MK2_Filter(44100)
rca.plot_freqz()

# plot_response(RCAFilter,[0.01, 0.25, 0.5, 0.75, 0.99],[1,2,5,10,100],)
# plt.grid()
# plt.show()

# vd = VoltageDivider(44100,1e6,1e6)
# vd.plot_freqz()

