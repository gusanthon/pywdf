from examples import *

# sweep knob positions of RCA mk2 SEF low pass filter
mk2_sef = rca_mk2_sef.RCA_MK2_SEF(44100, 0, 3000)
positions = range(1,11)
mk2_sef.plot_freqz_list(positions, mk2_sef.set_lowpass_knob_position, param_label = 'lpf knob pos')
 
# sweep resonance values in tr 808 hat resonator
hr = tr_808_hatresonator.TR_808_HatResonator(44100, 1000, .5)
resonances = [.1,.2,.3,.4,.5,.6,.7,.8,.9]
hr.plot_freqz_list(resonances, hr.set_resonance, param_label= 'resonance val')

# visualize transient response of Diode Clipper to AC signal
dc = diodeclipper.DiodeClipper(44100, cutoff= 1000, input_gain_db = 5)
dc.AC_transient_analysis()