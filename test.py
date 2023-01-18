from examples import *


sef = rca_mk2_sef.RCA_MK2_SEF(44100, 0, 3000)
positions = list(range(1,11))

# sweep positions of mk2 SEF low pass filter knob
sef.plot_freqz_list(positions, sef.set_lowpass_knob_position, param_label = 'lpf knob pos')


hr = tr_808_hatresonator.TR_808_HatResonator(44100, 1000, .5)
resonances = [.1,.2,.3,.4,.5,.6,.7,.8,.9]

# sweep resonance values in tr 808 hat resonator
hr.plot_freqz_list(resonances, hr.set_resonance)

