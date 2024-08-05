from pywdf import RCA_MK2_SEF, TR_808_HatResonator, DiodeClipper

# sweep positions of RCA mk2 SEF low pass filter knob and plot frequency responses
mk2_sef = RCA_MK2_SEF(44100, 0, 3000)
positions = range(0,11)
mk2_sef.plot_freqz_list(positions, mk2_sef.set_lowpass_knob_position, param_label = 'lpf knob pos', outpath='pywdf/figures/mk2_sef_lpf_knob.png')

# visualize impulse response of TR-808 hat resonator
hr = TR_808_HatResonator(44100, 1000, .5)
hr.plot_impulse_response(outpath='pywdf/figures/hat_res_IR.png')

# analyze transient response of Diode Clipper to AC signal
dc = DiodeClipper(44100, cutoff= 1000, input_gain_db = 5)
dc.AC_transient_analysis(outpath='pywdf/figures/diode_clipper_transient_anal.png')
