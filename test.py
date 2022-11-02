from matplotlib.pyplot import cm

from Circuit import *

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
hr = TR_808_HatResonator(44100,.5,.5)

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

plot_response(UnadaptedBaxandallEQ,[.5,.9],[0.01, 0.25, 0.5, 0.75, 0.99],plot='phase')
# plot_response(BaxandallEQ,[0.01, 0.25, 0.5, 0.75, 0.99],[0,.5,1])
# plot_response(TR_808_HatResonator,[1000,2000,3000,4000,5000,6000],[.01,.5,.1,])

# plot_response(PassiveLPF,[1000,2000,3000,4000,5000,6000])
plt.grid()
plt.show()

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

cd = CurrentDivider(44100,10000,10000)

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



class RCAFilter(Circuit):
    def __init__(self, sample_rate,L,C) -> None:
        self.L = L * 44.56e-3
        self.C = C * .402e-6
        self.fs = sample_rate

        self.Rt = Resistor(1e-9)
        self.L_LP4 = Inductor(L,self.fs)

        self.S8 = SeriesAdaptor(self.L_LP4, self.Rt)
        self.C_LP = Capacitor(C,self.fs)

        self.S7 = SeriesAdaptor(self.C_LP, self.S8)
        self.L_LP3 = Inductor(L,self.fs)

        self.S6 = SeriesAdaptor(self.L_LP3, self.S7)
        self.C_HP4 = Capacitor(C,self.fs)

        self.S4 = SeriesAdaptor(self.C_HP4,self.S6)
        self.L_HP2 = Inductor(L,self.fs)
        
        self.P2 = ParallelAdaptor(self.L_HP2,self.S4)
        self.C_HP3 = Capacitor(C,self.fs)

        self.S3 = SeriesAdaptor(self.C_HP3, self.P2)
        self.C_HP2 = Capacitor(C,self.fs)

        self.S2 = SeriesAdaptor(self.C_HP2,self.S3)
        self.L_LP1 = Inductor(L,self.fs)

        self.P1 = ParallelAdaptor(self.L_LP1,self.S2)
        self.C_HP1 = Capacitor(.272e-6,self.fs)

        self.S1 = SeriesAdaptor(self.C_HP1,self.P1)

        self.R0 = Resistor(1e10)
        self.S0 = SeriesAdaptor(self.S1,self.R0)
        self.Vin = IdealVoltageSource(self.S1)

        elements = [
            self.Rt,
            self.L_LP4,
            self.S8,
            self.C_LP, 
            self.S7,
            self.L_LP3,
            self.S6,
            self.C_HP4,
            self.S4,
            self.L_HP2,
            self.P2,
            self.C_HP3,
            self.S3,
            self.C_HP2,
            self.S2,
            self.L_LP1,
            self.P1,
            self.C_HP1,
            self.S1,
            self.Vin
        ]

        super().__init__(elements, self.Vin, self.Vin, self.Rt)


rca= RCAFilter(44100,1,1)
# rca.plot_freqz()

# plot_response(RCAFilter,[0.01, 0.25, 0.5, 0.75, 0.99],[1,2,5,10,100],)
# plt.grid()
# plt.show()

vd = VoltageDivider(44100,1e6,1e6)
vd.plot_freqz()

