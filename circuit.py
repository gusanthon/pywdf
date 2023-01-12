from wdf import *
from rtype import *
import scipy.signal
from scipy.io import wavfile
from scipy.fftpack import fft
import matplotlib.pyplot as plt


class Circuit:

    def __init__(self, source: baseWDF, root: baseWDF, output: baseWDF) -> None:
        """Initialize Circuit class functionality.

        Args:
            source (baseWDF): the circuit's voltage source
            root (baseWDF): the root of the wdf connection tree
            output (baseWDF): the component to be probed for output signal
        """
        self.source = source
        self.root = root
        self.output = output

    def process_sample(self, sample: float) -> float:
        """Process an individual sample with this circuit.

        Note: not every circuit will follow this general pattern, in such cases users may want to overwrite this function. See example circuits

        Args:
            sample (float): incoming sample to process

        Returns:
            float: processed sample
        """
        self.source.set_voltage(sample)
        self.root.accept_incident_wave(self.root.next.propagate_reflected_wave())
        self.root.next.accept_incident_wave(self.root.propagate_reflected_wave())
        return self.output.wave_to_voltage()

    def process_signal(self, signal: np.array) -> np.array:
        """Process an entire signal with this circuit.

        Args:
            signal (np.array): incoming signal to process

        Returns:
            np.array: processed signal
        """
        self.reset()
        return np.array([self.process_sample(sample) for sample in signal])

    def process_wav(self, filepath: str, output_filepath: str = None) -> np.array:
        fs, x = wavfile.read(filepath)
        if fs != self.fs:
            raise Exception(f'File sample rate differs from the {self.__class__.__name__}\'s')
        y = self.process_signal(x)
        if output_filepath is not None:
            wavfile.write(output_filepath,fs,y)
        return y

    def __call__(self, *args: any, **kwds: any) -> any:
        if isinstance(args[0], float) or isinstance(args[0], int):
            return self.process_sample(args[0])
        elif hasattr(args[0], "__iter__"):
            return self.process_signal(args[0])

    def get_impulse_response(self, delta_dur: float = 1, amp: float = 1) -> np.array:
        d = np.zeros(int(delta_dur * self.fs))
        d[0] = amp
        return self.process_signal(d)

    def set_sample_rate(self, new_fs: int) -> None:
        if self.fs != new_fs:
            self.fs = new_fs
            for key in self.__dict__:
                if hasattr(self.__dict__[key], 'fs'):
                    self.__dict__[key].prepare(new_fs)

    def reset(self) -> None:
        for key in self.__dict__:
            if isinstance(self.__dict__[key], baseWDF):
                self.__dict__[key].reset()

    def plot_freqz(self, outpath=None):
        x = self.get_impulse_response()
        #w, h = scipy.signal.freqz(x, 1, 2**15)
        fft_size = int(2 ** 15)
        nyquist = self.fs / 2
        N2 = int(fft_size / 2 - 1)
        h = fft(x, fft_size)[:N2]
        magnitude = 20 * np.log10(np.abs(h) + np.finfo(float).eps)
        phase = np.angle(h)
        magnitude_peak = np.max(magnitude)
        top_offset = 10
        bottom_offset = 70
        #frequencies = w / (2 * np.pi) * self.fs
        frequencies = np.linspace(0, nyquist, N2)

        # TODO: improve frequency axis
        _, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 6.5))
        xlims = [10**0, 10 ** np.log10(self.fs / 2)]
        ax[0].semilogx(frequencies, magnitude, label="WDF")
        ax[0].set_xlim(xlims)
        ax[0].set_ylim([magnitude_peak - bottom_offset, magnitude_peak + top_offset])
        ax[0].set_xlabel("Frequency [Hz]")
        ax[0].set_ylabel("Magnitude [dBFs]")
        ax[0].grid()
        ax[0].set_title(loc = 'left', label = self.__class__.__name__ + ' magnitude response')

        phase = 180 * phase / np.pi
        ax[1].semilogx(frequencies, phase,color='tab:orange')
        ax[1].set_xlim(xlims)
        ax[1].set_ylim([-180, 180])
        ax[1].set_xlabel("Frequency [Hz]")
        ax[1].set_ylabel("Phase [degrees]")
        ax[1].grid()
        ax[1].set_title(loc = 'left', label = self.__class__.__name__ + ' phase response')

        plt.tight_layout()
        if outpath:
            plt.savefig(outpath)
        plt.show()

    def plot_freqz_list(self, values:list, set_function:object, outpath:str=None):

        # TODO: create a legend with values
        # TODO: add a label as input parameter to be use in the legend
        # TODO: split this function in compute_magnitude_and_phase
        # plot_magnitude, plot_phase. In that way we can reuse the methods in plot_freqz()
        fft_size = int(2 ** 15)
        nyquist = self.fs / 2
        N2 = int(fft_size / 2 - 1)
        frequencies = np.linspace(0, nyquist, N2)

        _, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 6.5))

        for value in values:
            print(f"new value: {value}")
            set_function(value)
            x = self.get_impulse_response()

            h = fft(x, fft_size)[:N2]
            magnitude = 20 * np.log10(np.abs(h) + np.finfo(float).eps)
            phase = np.angle(h)
            magnitude_peak = np.max(magnitude)
            top_offset = 10
            bottom_offset = 70

            xlims = [10**0, 10 ** np.log10(self.fs / 2)]
            ax[0].semilogx(frequencies, magnitude, label="WDF")
            ax[0].set_xlim(xlims)
            ax[0].set_ylim([magnitude_peak - bottom_offset, magnitude_peak + top_offset])
            ax[0].set_xlabel("Frequency [Hz]")
            ax[0].set_ylabel("Magnitude [dBFs]")
            ax[0].grid()
            ax[0].set_title(loc = 'left', label = 'magnitude response')

            phase = 180 * phase / np.pi
            ax[1].semilogx(frequencies, phase,color='tab:orange')
            ax[1].set_xlim(xlims)
            ax[1].set_ylim([-180, 180])
            ax[1].set_xlabel("Frequency [Hz]")
            ax[1].set_ylabel("Phase [degrees]")
            ax[1].grid()
            ax[1].set_title(loc = 'left', label = ' phase response')

        plt.tight_layout()
        if outpath:
            plt.savefig(outpath)
        plt.show()


    def __impedance_calc(self, R: RTypeAdaptor):
        Ra, Rb, Rc, Rd, Re = R.get_port_impedances()
        R.set_S_matrix ([ [ -((Ra * Ra * Rb + Ra * Ra * Rc - Rb * Rc * Rc) * Rd * Rd - (Rb * Rb * Rc + Rb * Rc * Rc + Rb * Rd * Rd + (Rb * Rb + 2 * Rb * Rc) * Rd) * Re * Re + (Ra * Ra * Rb * Rb + 2 * Ra * Ra * Rb * Rc + (Ra * Ra - Rb * Rb) * Rc * Rc) * Rd + (Ra * Ra * Rb * Rb + 2 * Ra * Ra * Rb * Rc + (Ra * Ra - Rb * Rb) * Rc * Rc + (Ra * Ra - 2 * Rb * Rc) * Rd * Rd + 2 * (Ra * Ra * Rb - Rb * Rc * Rc + (Ra * Ra - Rb * Rb) * Rc) * Rd) * Re) / ((Ra * Ra * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb) * Rc) * Rd * Rd + (Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra + Rb) * Rd * Rd + (2 * Ra * Rb + Rb * Rb) * Rc + (2 * Ra * Rb + Rb * Rb + 2 * (Ra + Rb) * Rc) * Rd) * Re * Re + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc) * Rd + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb + 2 * (Ra + Rb) * Rc) * Rd * Rd + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 3 * Ra * Rb + Rb * Rb) * Rc) * Rd) * Re), -((Ra * Ra * Rc + Ra * Rc * Rc) * Rd * Rd + (Ra * Rb * Rc + Ra * Rc * Rc + Ra * Rd * Rd + (Ra * Rb + 2 * Ra * Rc) * Rd) * Re * Re + 2 * (Ra * Ra * Rb * Rc + (Ra * Ra + Ra * Rb) * Rc * Rc) * Rd + (2 * Ra * Ra * Rb * Rc + 2 * (Ra * Ra + Ra * Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rc) * Rd * Rd + (Ra * Ra * Rb + 2 * Ra * Rc * Rc + 3 * (Ra * Ra + Ra * Rb) * Rc) * Rd) * Re) / ((Ra * Ra * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb) * Rc) * Rd * Rd + (Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra + Rb) * Rd * Rd + (2 * Ra * Rb + Rb * Rb) * Rc + (2 * Ra * Rb + Rb * Rb + 2 * (Ra + Rb) * Rc) * Rd) * Re * Re + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc) * Rd + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb + 2 * (Ra + Rb) * Rc) * Rd * Rd + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 3 * Ra * Rb + Rb * Rb) * Rc) * Rd) * Re), ((2 * Ra * Ra * Rb + (Ra * Ra + 2 * Ra * Rb) * Rc) * Rd * Rd + (Ra * Rb * Rb + Ra * Rb * Rc + Ra * Rb * Rd) * Re * Re + 2 * (Ra * Ra * Rb * Rb + (Ra * Ra * Rb + Ra * Rb * Rb) * Rc) * Rd + (2 * Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb) * Rd * Rd + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc + (3 * Ra * Ra * Rb + 2 * Ra * Rb * Rb + (Ra * Ra + 3 * Ra * Rb) * Rc) * Rd) * Re) / ((Ra * Ra * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb) * Rc) * Rd * Rd + (Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra + Rb) * Rd * Rd + (2 * Ra * Rb + Rb * Rb) * Rc + (2 * Ra * Rb + Rb * Rb + 2 * (Ra + Rb) * Rc) * Rd) * Re * Re + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc) * Rd + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb + 2 * (Ra + Rb) * Rc) * Rd * Rd + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 3 * Ra * Rb + Rb * Rb) * Rc) * Rd) * Re), -((Ra * Rb * Rb + Ra * Rb * Rc + Ra * Rb * Rd) * Re * Re - (Ra * Ra * Rb * Rc + (Ra * Ra + Ra * Rb) * Rc * Rc) * Rd + (Ra * Ra * Rb * Rb + Ra * Rb * Rb * Rc - (Ra * Ra + Ra * Rb) * Rc * Rc + (Ra * Ra * Rb - Ra * Ra * Rc) * Rd) * Re) / ((Ra * Ra * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb) * Rc) * Rd * Rd + (Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra + Rb) * Rd * Rd + (2 * Ra * Rb + Rb * Rb) * Rc + (2 * Ra * Rb + Rb * Rb + 2 * (Ra + Rb) * Rc) * Rd) * Re * Re + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc) * Rd + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb + 2 * (Ra + Rb) * Rc) * Rd * Rd + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 3 * Ra * Rb + Rb * Rb) * Rc) * Rd) * Re), ((2 * Ra * Ra * Rb + (Ra * Ra + 2 * Ra * Rb) * Rc) * Rd * Rd + (2 * Ra * Ra * Rb * Rb + (Ra * Ra + Ra * Rb) * Rc * Rc + (3 * Ra * Ra * Rb + 2 * Ra * Rb * Rb) * Rc) * Rd + (Ra * Ra * Rb * Rb + (Ra * Ra + Ra * Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb) * Rd * Rd + (2 * Ra * Ra * Rb + Ra * Rb * Rb) * Rc + (2 * Ra * Ra * Rb + 2 * Ra * Rb * Rb + (2 * Ra * Ra + 3 * Ra * Rb) * Rc) * Rd) * Re) / ((Ra * Ra * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb) * Rc) * Rd * Rd + (Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra + Rb) * Rd * Rd + (2 * Ra * Rb + Rb * Rb) * Rc + (2 * Ra * Rb + Rb * Rb + 2 * (Ra + Rb) * Rc) * Rd) * Re * Re + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc) * Rd + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb + 2 * (Ra + Rb) * Rc) * Rd * Rd + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 3 * Ra * Rb + Rb * Rb) * Rc) * Rd) * Re), -(Ra * Rc * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re) / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re) ],
                    [ -((Ra * Rb * Rc + Rb * Rc * Rc) * Rd * Rd + (Rb * Rb * Rc + Rb * Rc * Rc + Rb * Rd * Rd + (Rb * Rb + 2 * Rb * Rc) * Rd) * Re * Re + 2 * (Ra * Rb * Rb * Rc + (Ra * Rb + Rb * Rb) * Rc * Rc) * Rd + (2 * Ra * Rb * Rb * Rc + 2 * (Ra * Rb + Rb * Rb) * Rc * Rc + (Ra * Rb + 2 * Rb * Rc) * Rd * Rd + (Ra * Rb * Rb + 2 * Rb * Rc * Rc + 3 * (Ra * Rb + Rb * Rb) * Rc) * Rd) * Re) / ((Ra * Ra * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb) * Rc) * Rd * Rd + (Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra + Rb) * Rd * Rd + (2 * Ra * Rb + Rb * Rb) * Rc + (2 * Ra * Rb + Rb * Rb + 2 * (Ra + Rb) * Rc) * Rd) * Re * Re + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc) * Rd + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb + 2 * (Ra + Rb) * Rc) * Rd * Rd + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 3 * Ra * Rb + Rb * Rb) * Rc) * Rd) * Re), ((Ra * Ra * Rc + Ra * Rc * Rc) * Rd * Rd - (Ra * Rb * Rb + Rb * Rb * Rc - Ra * Rc * Rc - Ra * Rd * Rd + (Rb * Rb - 2 * Ra * Rc) * Rd) * Re * Re - (Ra * Ra * Rb * Rb + 2 * Ra * Rb * Rb * Rc - (Ra * Ra - Rb * Rb) * Rc * Rc) * Rd - (Ra * Ra * Rb * Rb + 2 * Ra * Rb * Rb * Rc - (Ra * Ra - Rb * Rb) * Rc * Rc - (Ra * Ra + 2 * Ra * Rc) * Rd * Rd + 2 * (Ra * Rb * Rb - Ra * Rc * Rc - (Ra * Ra - Rb * Rb) * Rc) * Rd) * Re) / ((Ra * Ra * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb) * Rc) * Rd * Rd + (Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra + Rb) * Rd * Rd + (2 * Ra * Rb + Rb * Rb) * Rc + (2 * Ra * Rb + Rb * Rb + 2 * (Ra + Rb) * Rc) * Rd) * Re * Re + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc) * Rd + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb + 2 * (Ra + Rb) * Rc) * Rd * Rd + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 3 * Ra * Rb + Rb * Rb) * Rc) * Rd) * Re), -((Ra * Ra * Rb + Ra * Rb * Rc) * Rd * Rd + (2 * Ra * Rb * Rb + (2 * Ra * Rb + Rb * Rb) * Rc + (2 * Ra * Rb + Rb * Rb) * Rd) * Re * Re + 2 * (Ra * Ra * Rb * Rb + (Ra * Ra * Rb + Ra * Rb * Rb) * Rc) * Rd + (2 * Ra * Ra * Rb * Rb + Ra * Rb * Rd * Rd + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc + (2 * Ra * Ra * Rb + 3 * Ra * Rb * Rb + (3 * Ra * Rb + Rb * Rb) * Rc) * Rd) * Re) / ((Ra * Ra * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb) * Rc) * Rd * Rd + (Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra + Rb) * Rd * Rd + (2 * Ra * Rb + Rb * Rb) * Rc + (2 * Ra * Rb + Rb * Rb + 2 * (Ra + Rb) * Rc) * Rd) * Re * Re + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc) * Rd + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb + 2 * (Ra + Rb) * Rc) * Rd * Rd + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 3 * Ra * Rb + Rb * Rb) * Rc) * Rd) * Re), ((2 * Ra * Rb * Rb + (2 * Ra * Rb + Rb * Rb) * Rc + (2 * Ra * Rb + Rb * Rb) * Rd) * Re * Re + (Ra * Ra * Rb * Rb + (Ra * Rb + Rb * Rb) * Rc * Rc + (Ra * Ra * Rb + 2 * Ra * Rb * Rb) * Rc) * Rd + (2 * Ra * Ra * Rb * Rb + (Ra * Rb + Rb * Rb) * Rc * Rc + (2 * Ra * Ra * Rb + 3 * Ra * Rb * Rb) * Rc + (2 * Ra * Ra * Rb + 2 * Ra * Rb * Rb + (3 * Ra * Rb + 2 * Rb * Rb) * Rc) * Rd) * Re) / ((Ra * Ra * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb) * Rc) * Rd * Rd + (Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra + Rb) * Rd * Rd + (2 * Ra * Rb + Rb * Rb) * Rc + (2 * Ra * Rb + Rb * Rb + 2 * (Ra + Rb) * Rc) * Rd) * Re * Re + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc) * Rd + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb + 2 * (Ra + Rb) * Rc) * Rd * Rd + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 3 * Ra * Rb + Rb * Rb) * Rc) * Rd) * Re), -((Ra * Ra * Rb + Ra * Rb * Rc) * Rd * Rd + (Ra * Ra * Rb * Rb + Ra * Ra * Rb * Rc - (Ra * Rb + Rb * Rb) * Rc * Rc) * Rd - (Ra * Rb * Rb * Rc - Ra * Rb * Rd * Rd + (Ra * Rb + Rb * Rb) * Rc * Rc - (Ra * Rb * Rb - Rb * Rb * Rc) * Rd) * Re) / ((Ra * Ra * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb) * Rc) * Rd * Rd + (Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra + Rb) * Rd * Rd + (2 * Ra * Rb + Rb * Rb) * Rc + (2 * Ra * Rb + Rb * Rb + 2 * (Ra + Rb) * Rc) * Rd) * Re * Re + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc) * Rd + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb + 2 * (Ra + Rb) * Rc) * Rd * Rd + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 3 * Ra * Rb + Rb * Rb) * Rc) * Rd) * Re), -((Ra * Rb + Rb * Rc) * Rd + (Rb * Rc + Rb * Rd) * Re) / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re) ],
                    [ ((2 * Ra * Rb * Rc + (Ra + 2 * Rb) * Rc * Rc) * Rd * Rd + (Rb * Rb * Rc + Rb * Rc * Rc + Rb * Rc * Rd) * Re * Re + 2 * (Ra * Rb * Rb * Rc + (Ra * Rb + Rb * Rb) * Rc * Rc) * Rd + (2 * Ra * Rb * Rb * Rc + (Ra + 2 * Rb) * Rc * Rd * Rd + 2 * (Ra * Rb + Rb * Rb) * Rc * Rc + ((Ra + 3 * Rb) * Rc * Rc + (3 * Ra * Rb + 2 * Rb * Rb) * Rc) * Rd) * Re) / ((Ra * Ra * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb) * Rc) * Rd * Rd + (Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra + Rb) * Rd * Rd + (2 * Ra * Rb + Rb * Rb) * Rc + (2 * Ra * Rb + Rb * Rb + 2 * (Ra + Rb) * Rc) * Rd) * Re * Re + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc) * Rd + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb + 2 * (Ra + Rb) * Rc) * Rd * Rd + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 3 * Ra * Rb + Rb * Rb) * Rc) * Rd) * Re), -((Ra * Ra * Rc + Ra * Rc * Rc) * Rd * Rd + (2 * Ra * Rb * Rc + (2 * Ra + Rb) * Rc * Rc + (2 * Ra + Rb) * Rc * Rd) * Re * Re + 2 * (Ra * Ra * Rb * Rc + (Ra * Ra + Ra * Rb) * Rc * Rc) * Rd + (2 * Ra * Ra * Rb * Rc + Ra * Rc * Rd * Rd + 2 * (Ra * Ra + Ra * Rb) * Rc * Rc + ((3 * Ra + Rb) * Rc * Rc + (2 * Ra * Ra + 3 * Ra * Rb) * Rc) * Rd) * Re) / ((Ra * Ra * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb) * Rc) * Rd * Rd + (Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra + Rb) * Rd * Rd + (2 * Ra * Rb + Rb * Rb) * Rc + (2 * Ra * Rb + Rb * Rb + 2 * (Ra + Rb) * Rc) * Rd) * Re * Re + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc) * Rd + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb + 2 * (Ra + Rb) * Rc) * Rd * Rd + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 3 * Ra * Rb + Rb * Rb) * Rc) * Rd) * Re), ((Ra * Ra * Rb - (Ra + Rb) * Rc * Rc) * Rd * Rd + (Ra * Rb * Rb - (Ra + Rb) * Rc * Rc + (Ra + Rb) * Rd * Rd + (2 * Ra * Rb + Rb * Rb) * Rd) * Re * Re + (Ra * Ra * Rb * Rb - (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc) * Rd + (Ra * Ra * Rb * Rb - (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb) * Rd * Rd + 2 * (Ra * Ra * Rb + Ra * Rb * Rb - (Ra + Rb) * Rc * Rc) * Rd) * Re) / ((Ra * Ra * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb) * Rc) * Rd * Rd + (Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra + Rb) * Rd * Rd + (2 * Ra * Rb + Rb * Rb) * Rc + (2 * Ra * Rb + Rb * Rb + 2 * (Ra + Rb) * Rc) * Rd) * Re * Re + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc) * Rd + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb + 2 * (Ra + Rb) * Rc) * Rd * Rd + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 3 * Ra * Rb + Rb * Rb) * Rc) * Rd) * Re), ((2 * (Ra + Rb) * Rc * Rc + 2 * (Ra + Rb) * Rc * Rd + (2 * Ra * Rb + Rb * Rb) * Rc) * Re * Re + (Ra * Ra * Rb * Rc + (Ra * Ra + Ra * Rb) * Rc * Rc) * Rd + ((2 * Ra * Ra + 3 * Ra * Rb + Rb * Rb) * Rc * Rc + (2 * Ra * Ra * Rb + Ra * Rb * Rb) * Rc + (2 * (Ra + Rb) * Rc * Rc + (2 * Ra * Ra + 3 * Ra * Rb) * Rc) * Rd) * Re) / ((Ra * Ra * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb) * Rc) * Rd * Rd + (Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra + Rb) * Rd * Rd + (2 * Ra * Rb + Rb * Rb) * Rc + (2 * Ra * Rb + Rb * Rb + 2 * (Ra + Rb) * Rc) * Rd) * Re * Re + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc) * Rd + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb + 2 * (Ra + Rb) * Rc) * Rd * Rd + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 3 * Ra * Rb + Rb * Rb) * Rc) * Rd) * Re), -((2 * (Ra + Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb) * Rc) * Rd * Rd + ((Ra * Ra + 3 * Ra * Rb + 2 * Rb * Rb) * Rc * Rc + (Ra * Ra * Rb + 2 * Ra * Rb * Rb) * Rc) * Rd + (Ra * Rb * Rb * Rc + 2 * (Ra + Rb) * Rc * Rd * Rd + (Ra * Rb + Rb * Rb) * Rc * Rc + (2 * (Ra + Rb) * Rc * Rc + (3 * Ra * Rb + 2 * Rb * Rb) * Rc) * Rd) * Re) / ((Ra * Ra * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb) * Rc) * Rd * Rd + (Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra + Rb) * Rd * Rd + (2 * Ra * Rb + Rb * Rb) * Rc + (2 * Ra * Rb + Rb * Rb + 2 * (Ra + Rb) * Rc) * Rd) * Re * Re + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc) * Rd + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb + 2 * (Ra + Rb) * Rc) * Rd * Rd + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 3 * Ra * Rb + Rb * Rb) * Rc) * Rd) * Re), -(Ra * Rc * Rd - Rb * Rc * Re) / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re) ],
                    [ ((Ra * Rb * Rc + (Ra + Rb) * Rc * Rc) * Rd * Rd - (Rb * Rd * Rd + (Rb * Rb + Rb * Rc) * Rd) * Re * Re - ((Ra * Rb - Ra * Rc) * Rd * Rd + (Ra * Rb * Rb + Rb * Rb * Rc - (Ra + Rb) * Rc * Rc) * Rd) * Re) / ((Ra * Ra * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb) * Rc) * Rd * Rd + (Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra + Rb) * Rd * Rd + (2 * Ra * Rb + Rb * Rb) * Rc + (2 * Ra * Rb + Rb * Rb + 2 * (Ra + Rb) * Rc) * Rd) * Re * Re + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc) * Rd + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb + 2 * (Ra + Rb) * Rc) * Rd * Rd + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 3 * Ra * Rb + Rb * Rb) * Rc) * Rd) * Re), ((Ra * Ra * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb) * Rc) * Rd * Rd + ((2 * Ra + Rb) * Rd * Rd + (2 * Ra * Rb + (2 * Ra + Rb) * Rc) * Rd) * Re * Re + ((2 * Ra * Ra + 2 * Ra * Rb + (3 * Ra + 2 * Rb) * Rc) * Rd * Rd + (2 * Ra * Ra * Rb + (Ra + Rb) * Rc * Rc + (2 * Ra * Ra + 3 * Ra * Rb) * Rc) * Rd) * Re) / ((Ra * Ra * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb) * Rc) * Rd * Rd + (Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra + Rb) * Rd * Rd + (2 * Ra * Rb + Rb * Rb) * Rc + (2 * Ra * Rb + Rb * Rb + 2 * (Ra + Rb) * Rc) * Rd) * Re * Re + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc) * Rd + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb + 2 * (Ra + Rb) * Rc) * Rd * Rd + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 3 * Ra * Rb + Rb * Rb) * Rc) * Rd) * Re), ((Ra * Ra * Rb + (Ra * Ra + Ra * Rb) * Rc) * Rd * Rd + (2 * (Ra + Rb) * Rd * Rd + (2 * Ra * Rb + Rb * Rb + 2 * (Ra + Rb) * Rc) * Rd) * Re * Re + ((2 * Ra * Ra + 3 * Ra * Rb + 2 * (Ra + Rb) * Rc) * Rd * Rd + (2 * Ra * Ra * Rb + Ra * Rb * Rb + (2 * Ra * Ra + 3 * Ra * Rb + Rb * Rb) * Rc) * Rd) * Re) / ((Ra * Ra * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb) * Rc) * Rd * Rd + (Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra + Rb) * Rd * Rd + (2 * Ra * Rb + Rb * Rb) * Rc + (2 * Ra * Rb + Rb * Rb + 2 * (Ra + Rb) * Rc) * Rd) * Re * Re + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc) * Rd + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb + 2 * (Ra + Rb) * Rc) * Rd * Rd + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 3 * Ra * Rb + Rb * Rb) * Rc) * Rd) * Re), -((Ra * Ra * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb) * Rc) * Rd * Rd - (Ra * Rb * Rb + (Ra + Rb) * Rc * Rc - (Ra + Rb) * Rd * Rd + (2 * Ra * Rb + Rb * Rb) * Rc) * Re * Re - (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc - (Ra * Ra + 2 * Ra * Rb + 2 * (Ra + Rb) * Rc) * Rd * Rd + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc) * Re) / ((Ra * Ra * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb) * Rc) * Rd * Rd + (Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra + Rb) * Rd * Rd + (2 * Ra * Rb + Rb * Rb) * Rc + (2 * Ra * Rb + Rb * Rb + 2 * (Ra + Rb) * Rc) * Rd) * Re * Re + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc) * Rd + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb + 2 * (Ra + Rb) * Rc) * Rd * Rd + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 3 * Ra * Rb + Rb * Rb) * Rc) * Rd) * Re), -((Ra * Ra * Rb + 2 * (Ra + Rb) * Rc * Rc + (Ra * Ra + 3 * Ra * Rb) * Rc) * Rd * Rd + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc) * Rd + ((Ra * Rb + 2 * (Ra + Rb) * Rc) * Rd * Rd + (Ra * Rb * Rb + 2 * (Ra + Rb) * Rc * Rc + (3 * Ra * Rb + Rb * Rb) * Rc) * Rd) * Re) / ((Ra * Ra * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb) * Rc) * Rd * Rd + (Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra + Rb) * Rd * Rd + (2 * Ra * Rb + Rb * Rb) * Rc + (2 * Ra * Rb + Rb * Rb + 2 * (Ra + Rb) * Rc) * Rd) * Re * Re + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc) * Rd + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb + 2 * (Ra + Rb) * Rc) * Rd * Rd + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 3 * Ra * Rb + Rb * Rb) * Rc) * Rd) * Re), -(Rb * Rd * Re + (Ra * Rb + (Ra + Rb) * Rc) * Rd) / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re) ],
                    [ ((Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra + 2 * Rb) * Rd * Rd + (2 * Ra * Rb + Rb * Rb) * Rc + (2 * Ra * Rb + 2 * Rb * Rb + (2 * Ra + 3 * Rb) * Rc) * Rd) * Re * Re + ((2 * Ra * Rb + (Ra + 2 * Rb) * Rc) * Rd * Rd + (2 * Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (3 * Ra * Rb + 2 * Rb * Rb) * Rc) * Rd) * Re) / ((Ra * Ra * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb) * Rc) * Rd * Rd + (Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra + Rb) * Rd * Rd + (2 * Ra * Rb + Rb * Rb) * Rc + (2 * Ra * Rb + Rb * Rb + 2 * (Ra + Rb) * Rc) * Rd) * Re * Re + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc) * Rd + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb + 2 * (Ra + Rb) * Rc) * Rd * Rd + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 3 * Ra * Rb + Rb * Rb) * Rc) * Rd) * Re), ((Ra * Rb * Rc + (Ra + Rb) * Rc * Rc - Ra * Rd * Rd - (Ra * Rb - Rb * Rc) * Rd) * Re * Re - ((Ra * Ra + Ra * Rc) * Rd * Rd + (Ra * Ra * Rb + Ra * Ra * Rc - (Ra + Rb) * Rc * Rc) * Rd) * Re) / ((Ra * Ra * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb) * Rc) * Rd * Rd + (Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra + Rb) * Rd * Rd + (2 * Ra * Rb + Rb * Rb) * Rc + (2 * Ra * Rb + Rb * Rb + 2 * (Ra + Rb) * Rc) * Rd) * Re * Re + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc) * Rd + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb + 2 * (Ra + Rb) * Rc) * Rd * Rd + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 3 * Ra * Rb + Rb * Rb) * Rc) * Rd) * Re), -((Ra * Rb * Rb + 2 * (Ra + Rb) * Rd * Rd + (Ra * Rb + Rb * Rb) * Rc + (3 * Ra * Rb + 2 * Rb * Rb + 2 * (Ra + Rb) * Rc) * Rd) * Re * Re + ((Ra * Ra + 2 * Ra * Rb + 2 * (Ra + Rb) * Rc) * Rd * Rd + (Ra * Ra * Rb + 2 * Ra * Rb * Rb + (Ra * Ra + 3 * Ra * Rb + 2 * Rb * Rb) * Rc) * Rd) * Re) / ((Ra * Ra * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb) * Rc) * Rd * Rd + (Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra + Rb) * Rd * Rd + (2 * Ra * Rb + Rb * Rb) * Rc + (2 * Ra * Rb + Rb * Rb + 2 * (Ra + Rb) * Rc) * Rd) * Re * Re + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc) * Rd + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb + 2 * (Ra + Rb) * Rc) * Rd * Rd + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 3 * Ra * Rb + Rb * Rb) * Rc) * Rd) * Re), -((Ra * Rb * Rb + 2 * (Ra + Rb) * Rc * Rc + (3 * Ra * Rb + Rb * Rb) * Rc + (Ra * Rb + 2 * (Ra + Rb) * Rc) * Rd) * Re * Re + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc + (Ra * Ra * Rb + 2 * (Ra + Rb) * Rc * Rc + (Ra * Ra + 3 * Ra * Rb) * Rc) * Rd) * Re) / ((Ra * Ra * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb) * Rc) * Rd * Rd + (Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra + Rb) * Rd * Rd + (2 * Ra * Rb + Rb * Rb) * Rc + (2 * Ra * Rb + Rb * Rb + 2 * (Ra + Rb) * Rc) * Rd) * Re * Re + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc) * Rd + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb + 2 * (Ra + Rb) * Rc) * Rd * Rd + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 3 * Ra * Rb + Rb * Rb) * Rc) * Rd) * Re), ((Ra * Ra * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb) * Rc) * Rd * Rd - (Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra + Rb) * Rd * Rd + (2 * Ra * Rb + Rb * Rb) * Rc + (2 * Ra * Rb + Rb * Rb + 2 * (Ra + Rb) * Rc) * Rd) * Re * Re + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc) * Rd) / ((Ra * Ra * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb) * Rc) * Rd * Rd + (Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra + Rb) * Rd * Rd + (2 * Ra * Rb + Rb * Rb) * Rc + (2 * Ra * Rb + Rb * Rb + 2 * (Ra + Rb) * Rc) * Rd) * Re * Re + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc) * Rd + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb + 2 * (Ra + Rb) * Rc) * Rd * Rd + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 3 * Ra * Rb + Rb * Rb) * Rc) * Rd) * Re), -(Ra * Rb + (Ra + Rb) * Rc + Ra * Rd) * Re / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re) ],
                    [ -(Rc * Rd + (Rb + Rc + Rd) * Re) / (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re), -((Ra + Rc) * Rd + (Rc + Rd) * Re) / (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re), -(Ra * Rd - Rb * Re) / (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re), -(Ra * Rb + (Ra + Rb) * Rc + Rb * Re) / (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re), -(Ra * Rb + (Ra + Rb) * Rc + Ra * Rd) / (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re), 0 ] ])
        Rf = ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re) / (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re)
        return Rf

    def set_bass(self, new_bass: float) -> None:
        if self.bass != new_bass:
            if new_bass <= 0:
                new_bass = 1e-20
            elif new_bass >= 1:
                new_bass = .99999999999999
            self.Pb_plus.set_resistance(self.Pb * new_bass)
            self.Pb_minus.set_resistance(self.Pb * (1 - new_bass))
            self.bass = new_bass

    def set_treble(self, new_treble: float) -> None:
        if self.treble != new_treble:
            if new_treble <= 0:
                new_treble = 1e-20
            elif new_treble >= 1:
                new_treble = .99999999999999
            self.Pt_plus.set_resistance(self.Pt * new_treble)
            self.Pt_plus.set_resistance(self.Pt * (1 - new_treble))
            self.treble = new_treble


############################################################################################


class UnadaptedBaxandallEQ(BaxandallEQ):
    def __init__(self, fs: int, bass: float, treble: float) -> None:

        def __impedance_calc(R: RootRTypeAdaptor):
            Ra, Rb, Rc, Rd, Re, Rf = R.get_port_impedances()
            R.set_S_matrix ([ [ -2 * ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Ra / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf) + 1, -2 * (Rc * Rd + (Rc + Rd) * Re + Rc * Rf) * Ra / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf), 2 * (Rb * Rd + Rb * Re + (Rb + Rd) * Rf) * Ra / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf), -2 * (Rb * Re - Rc * Rf) * Ra / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf), 2 * (Rb * Rd + (Rb + Rc + Rd) * Rf) * Ra / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf), -2 * (Rc * Rd + (Rb + Rc + Rd) * Re) * Ra / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf) ],
                    [ -2 * (Rc * Rd + (Rc + Rd) * Re + Rc * Rf) * Rb / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf), -2 * ((Ra + Rc) * Rd + (Ra + Rc + Rd) * Re + (Ra + Rc + Re) * Rf) * Rb / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf) + 1, -2 * (Ra * Rd + Ra * Re + (Ra + Re) * Rf) * Rb / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf), 2 * (Ra * Re + (Ra + Rc + Re) * Rf) * Rb / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf), -2 * (Ra * Rd - Rc * Rf) * Rb / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf), -2 * ((Ra + Rc) * Rd + (Rc + Rd) * Re) * Rb / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf) ],
                    [ 2 * (Rb * Rd + Rb * Re + (Rb + Rd) * Rf) * Rc / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf), -2 * (Ra * Rd + Ra * Re + (Ra + Re) * Rf) * Rc / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf), -2 * ((Ra + Rb) * Rd + (Ra + Rb) * Re + (Ra + Rb + Rd + Re) * Rf) * Rc / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf) + 1, 2 * ((Ra + Rb) * Re + (Ra + Re) * Rf) * Rc / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf), -2 * ((Ra + Rb) * Rd + (Rb + Rd) * Rf) * Rc / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf), -2 * (Ra * Rd - Rb * Re) * Rc / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf) ],
                    [ -2 * (Rb * Re - Rc * Rf) * Rd / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf), 2 * (Ra * Re + (Ra + Rc + Re) * Rf) * Rd / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf), 2 * ((Ra + Rb) * Re + (Ra + Re) * Rf) * Rd / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf), -2 * (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Re + (Ra + Rc + Re) * Rf) * Rd / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf) + 1, -2 * (Ra * Rb + (Ra + Rb) * Rc + Rc * Rf) * Rd / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf), -2 * (Ra * Rb + (Ra + Rb) * Rc + Rb * Re) * Rd / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf) ],
                    [ 2 * (Rb * Rd + (Rb + Rc + Rd) * Rf) * Re / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf), -2 * (Ra * Rd - Rc * Rf) * Re / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf), -2 * ((Ra + Rb) * Rd + (Rb + Rd) * Rf) * Re / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf), -2 * (Ra * Rb + (Ra + Rb) * Rc + Rc * Rf) * Re / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf), -2 * (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd + (Rb + Rc + Rd) * Rf) * Re / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf) + 1, -2 * (Ra * Rb + (Ra + Rb) * Rc + Ra * Rd) * Re / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf) ],
                    [ -2 * (Rc * Rd + (Rb + Rc + Rd) * Re) * Rf / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf), -2 * ((Ra + Rc) * Rd + (Rc + Rd) * Re) * Rf / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf), -2 * (Ra * Rd - Rb * Re) * Rf / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf), -2 * (Ra * Rb + (Ra + Rb) * Rc + Rb * Re) * Rf / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf), -2 * (Ra * Rb + (Ra + Rb) * Rc + Ra * Rd) * Rf / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf), -2 * (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf) + 1 ] ])

        super().__init__(fs, bass, treble)

        self.Vin = ResistiveVoltageSource()
        self.S1 = SeriesAdaptor(self.Vin,self.Ca)
        self.R_adaptor = RootRTypeAdaptor([self.S4,self.P1,self.Resc, self.S3, self.S2, self.S1], __impedance_calc)

        self.elements = [
            self.Pt_plus, self.Resd, self.P4, self.Cd, self.S4, self.Pt_minus, self.Rese, self.P5, self.Ce, self.S5,
            self.Rl, self.P1, self.Resc, self.Pb_minus, self.Cc, self.P3, self.Resb, self.S3, self.Pb_plus, self.Cb,
            self.P2, self.Resa, self.S2, self.R_adaptor, self.Ca, self.S1, self.Vin
        ]

        self.root = self.R_adaptor
        self.source = self.Vin

    def process_sample(self, sample: float) -> float:
        self.Vin.set_voltage(sample)
        self.R_adaptor.compute()
        return self.output.wave_to_voltage()


############################################################################################


class TR_808_HatResonator(Circuit):
    def __init__(self, fs: int, cutoff: float, resonance: float) -> None:

        self.fs = fs
        self.cutoff = cutoff
        self.resonance = resonance

        self.Vin = ResistiveVoltageSource(22e3)
        self.C4 = Capacitor(1e-9, self.fs)
        self.S1 = SeriesAdaptor(self.Vin, self.C4)

        self.R197 = Resistor(820e3)
        self.C58 = Capacitor(.027e-6, self.fs)
        self.C59 = Capacitor(.027e-6, self.fs)
        self.R196 = Resistor(680)
        self.R_adaptor = RootRTypeAdaptor([self.S1, self.R197, self.C58, self.C59, self.R196], self.__impedance_calc)

        elements = [
            self.Vin,
            self.C4,
            self.S1,
            self.R197,
            self.C58,
            self.C59,
            self.R196,
            self.R_adaptor
        ]

        self.__set_components()

        super().__init__(elements, self.Vin, self.R_adaptor, self.R196)


    def process_sample(self, sample: float) -> float:
        self.Vin.set_voltage(-sample)
        self.R_adaptor.compute()
        return self.output.wave_to_voltage() + self.C59.wave_to_voltage()

    def __set_components(self) -> None:
        Rfb = 82e3
        R_g = 10000 ** ((1 - self.resonance) ** 0.37)
        C = 1 / (2 * np.pi * self.cutoff * np.sqrt(Rfb * R_g))
        self.R197.set_resistance(Rfb)
        self.R196.set_resistance(R_g)
        self.C58.set_capacitance(C)
        self.C59.set_capacitance(C)

    def set_cutoff(self, new_cutoff: float) -> None:
        if self.cutoff != new_cutoff:
            self.cutoff = new_cutoff
            self.__set_components()

    def set_resonance(self, new_res: float) -> None:
        if self.resonance != new_res:
            self.resonance == new_res
            self.__set_components()

    def __impedance_calc(self, R: RootRTypeAdaptor) -> None:
        Ag = 100
        Ri = 1e9
        Ro = 1e-1
        Ra, Rb, Rc, Rd, Re = R.get_port_impedances()
        R.set_S_matrix ([ [ -((Ra * Rb + (Ra - Rb) * Rc) * Rd + (Ra * Rb + (Ra - Rb) * Rc + (Ra - Rb) * Rd) * Re - (((Ag + 1) * Rc + Rb) * Rd + ((Ag + 1) * Rb + (Ag + 1) * Rc + (Ag + 1) * Rd) * Re) * Ri - (Ra * Rb + (Ra - Rb) * Rc + (Ra - Rc) * Rd - (Rb + Rc + Rd) * Re - (Rb + Rc + Rd) * Ri) * Ro) / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (((Ag + 1) * Rc + Rb) * Rd + ((Ag + 1) * Rb + (Ag + 1) * Rc + (Ag + 1) * Rd) * Re) * Ri - (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Ri) * Ro), -2 * (Ra * Rc * Rd - Ra * Rc * Ro + (Ra * Rc + Ra * Rd) * Re) / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (((Ag + 1) * Rc + Rb) * Rd + ((Ag + 1) * Rb + (Ag + 1) * Rc + (Ag + 1) * Rd) * Re) * Ri - (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Ri) * Ro), 2 * (Ra * Rb * Rd + Ra * Rb * Re - (Ra * Rb + Ra * Rd) * Ro) / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (((Ag + 1) * Rc + Rb) * Rd + ((Ag + 1) * Rb + (Ag + 1) * Rc + (Ag + 1) * Rd) * Re) * Ri - (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Ri) * Ro), 2 * (Ra * Rb * Re + Ra * Rc * Ro) / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (((Ag + 1) * Rc + Rb) * Rd + ((Ag + 1) * Rb + (Ag + 1) * Rc + (Ag + 1) * Rd) * Re) * Ri - (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Ri) * Ro), 2 * (Ra * Rb * Rd - (Ra * Rb + Ra * Rc + Ra * Rd) * Ro) / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (((Ag + 1) * Rc + Rb) * Rd + ((Ag + 1) * Rb + (Ag + 1) * Rc + (Ag + 1) * Rd) * Re) * Ri - (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Ri) * Ro) ],
                            [ 2 * (Ag * Rb * Rd * Ri - Rb * Rc * Rd + Rb * Rc * Ro - (Rb * Rc + Rb * Rd) * Re) / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (((Ag + 1) * Rc + Rb) * Rd + ((Ag + 1) * Rb + (Ag + 1) * Rc + (Ag + 1) * Rd) * Re) * Ri - (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Ri) * Ro), -((Ra * Rb - (Ra - Rb) * Rc) * Rd + (Ra * Rb - (Ra - Rb) * Rc - (Ra - Rb) * Rd) * Re - (((Ag + 1) * Rc - Rb) * Rd - ((Ag + 1) * Rb - (Ag + 1) * Rc - (Ag + 1) * Rd) * Re) * Ri - (Ra * Rb - (Ra - Rb) * Rc - (Ra + Rc) * Rd + (Rb - Rc - Rd) * Re + (Rb - Rc - Rd) * Ri) * Ro) / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (((Ag + 1) * Rc + Rb) * Rd + ((Ag + 1) * Rb + (Ag + 1) * Rc + (Ag + 1) * Rd) * Re) * Ri - (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Ri) * Ro), -2 * (Ra * Rb * Rd + Ra * Rb * Re + ((Ag + 1) * Rb * Rd + (Ag + 1) * Rb * Re) * Ri - (Ra * Rb + Rb * Re + Rb * Ri) * Ro) / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (((Ag + 1) * Rc + Rb) * Rd + ((Ag + 1) * Rb + (Ag + 1) * Rc + (Ag + 1) * Rd) * Re) * Ri - (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Ri) * Ro), -2 * ((Ag + 1) * Rb * Re * Ri + Ra * Rb * Re - (Ra * Rb + Rb * Rc + Rb * Re + Rb * Ri) * Ro) / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (((Ag + 1) * Rc + Rb) * Rd + ((Ag + 1) * Rb + (Ag + 1) * Rc + (Ag + 1) * Rd) * Re) * Ri - (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Ri) * Ro), -2 * ((Ag + 1) * Rb * Rd * Ri + Ra * Rb * Rd + Rb * Rc * Ro) / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (((Ag + 1) * Rc + Rb) * Rd + ((Ag + 1) * Rb + (Ag + 1) * Rc + (Ag + 1) * Rd) * Re) * Ri - (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Ri) * Ro) ],
                            [ 2 * (Ag * Rc * Rd * Ri + Rb * Rc * Rd + Rb * Rc * Re - (Rb * Rc + Rc * Rd) * Ro) / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (((Ag + 1) * Rc + Rb) * Rd + ((Ag + 1) * Rb + (Ag + 1) * Rc + (Ag + 1) * Rd) * Re) * Ri - (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Ri) * Ro), -2 * (Ra * Rc * Rd + Ra * Rc * Re + ((Ag + 1) * Rc * Re + Rc * Rd) * Ri - (Ra * Rc + Rc * Re + Rc * Ri) * Ro) / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (((Ag + 1) * Rc + Rb) * Rd + ((Ag + 1) * Rb + (Ag + 1) * Rc + (Ag + 1) * Rd) * Re) * Ri - (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Ri) * Ro), ((Ra * Rb - (Ra + Rb) * Rc) * Rd + (Ra * Rb - (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re - (((Ag + 1) * Rc - Rb) * Rd - ((Ag + 1) * Rb - (Ag + 1) * Rc + (Ag + 1) * Rd) * Re) * Ri - (Ra * Rb - (Ra + Rb) * Rc + (Ra - Rc) * Rd + (Rb - Rc + Rd) * Re + (Rb - Rc + Rd) * Ri) * Ro) / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (((Ag + 1) * Rc + Rb) * Rd + ((Ag + 1) * Rb + (Ag + 1) * Rc + (Ag + 1) * Rd) * Re) * Ri - (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Ri) * Ro), -2 * ((Ag + 1) * Rc * Re * Ri + (Ra + Rb) * Rc * Re - (Ra * Rc + Rc * Re + Rc * Ri) * Ro) / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (((Ag + 1) * Rc + Rb) * Rd + ((Ag + 1) * Rb + (Ag + 1) * Rc + (Ag + 1) * Rd) * Re) * Ri - (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Ri) * Ro), -2 * ((Ag + 1) * Rc * Rd * Ri + (Ra + Rb) * Rc * Rd - (Rb * Rc + Rc * Rd) * Ro) / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (((Ag + 1) * Rc + Rb) * Rd + ((Ag + 1) * Rb + (Ag + 1) * Rc + (Ag + 1) * Rd) * Re) * Ri - (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Ri) * Ro) ],
                            [ 2 * (Rb * Rd * Re - (Ag * Rb + Ag * Rc) * Rd * Ri + Rc * Rd * Ro) / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (((Ag + 1) * Rc + Rb) * Rd + ((Ag + 1) * Rb + (Ag + 1) * Rc + (Ag + 1) * Rd) * Re) * Ri - (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Ri) * Ro), -2 * (Ra * Rd * Re + (Ag * Rc * Rd + (Ag + 1) * Rd * Re) * Ri - ((Ra + Rc) * Rd + Rd * Re + Rd * Ri) * Ro) / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (((Ag + 1) * Rc + Rb) * Rd + ((Ag + 1) * Rb + (Ag + 1) * Rc + (Ag + 1) * Rd) * Re) * Ri - (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Ri) * Ro), -2 * ((Ra + Rb) * Rd * Re - (Ag * Rb * Rd - (Ag + 1) * Rd * Re) * Ri - (Ra * Rd + Rd * Re + Rd * Ri) * Ro) / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (((Ag + 1) * Rc + Rb) * Rd + ((Ag + 1) * Rb + (Ag + 1) * Rc + (Ag + 1) * Rd) * Re) * Ri - (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Ri) * Ro), -((Ra * Rb + (Ra + Rb) * Rc) * Rd - (Ra * Rb + (Ra + Rb) * Rc - (Ra + Rb) * Rd) * Re + (((Ag + 1) * Rc + Rb) * Rd - ((Ag + 1) * Rb + (Ag + 1) * Rc - (Ag + 1) * Rd) * Re) * Ri + (Ra * Rb + (Ra + Rb) * Rc - (Ra + Rc) * Rd + (Rb + Rc - Rd) * Re + (Rb + Rc - Rd) * Ri) * Ro) / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (((Ag + 1) * Rc + Rb) * Rd + ((Ag + 1) * Rb + (Ag + 1) * Rc + (Ag + 1) * Rd) * Re) * Ri - (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Ri) * Ro), 2 * (((Ag + 1) * Rb + (Ag + 1) * Rc) * Rd * Ri - Rc * Rd * Ro + (Ra * Rb + (Ra + Rb) * Rc) * Rd) / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (((Ag + 1) * Rc + Rb) * Rd + ((Ag + 1) * Rb + (Ag + 1) * Rc + (Ag + 1) * Rd) * Re) * Ri - (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Ri) * Ro) ],
                            [ 2 * (Rb * Rd * Re + (Ag * Rb + Ag * Rc + Ag * Rd) * Re * Ri - (Rb + Rc + Rd) * Re * Ro) / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (((Ag + 1) * Rc + Rb) * Rd + ((Ag + 1) * Rb + (Ag + 1) * Rc + (Ag + 1) * Rd) * Re) * Ri - (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Ri) * Ro), -2 * (Ra * Rd * Re - (Ag * Rc - Rd) * Re * Ri + Rc * Re * Ro) / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (((Ag + 1) * Rc + Rb) * Rd + ((Ag + 1) * Rb + (Ag + 1) * Rc + (Ag + 1) * Rd) * Re) * Ri - (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Ri) * Ro), -2 * ((Ra + Rb) * Rd * Re + (Ag * Rb + (Ag + 1) * Rd) * Re * Ri - (Rb + Rd) * Re * Ro) / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (((Ag + 1) * Rc + Rb) * Rd + ((Ag + 1) * Rb + (Ag + 1) * Rc + (Ag + 1) * Rd) * Re) * Ri - (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Ri) * Ro), 2 * (((Ag + 1) * Rc + Rb) * Re * Ri - Rc * Re * Ro + (Ra * Rb + (Ra + Rb) * Rc) * Re) / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (((Ag + 1) * Rc + Rb) * Rd + ((Ag + 1) * Rb + (Ag + 1) * Rc + (Ag + 1) * Rd) * Re) * Ri - (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Ri) * Ro), ((Ra * Rb + (Ra + Rb) * Rc) * Rd - (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (((Ag + 1) * Rc + Rb) * Rd - ((Ag + 1) * Rb + (Ag + 1) * Rc + (Ag + 1) * Rd) * Re) * Ri - (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd - (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Ri) * Ro) / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (((Ag + 1) * Rc + Rb) * Rd + ((Ag + 1) * Rb + (Ag + 1) * Rc + (Ag + 1) * Rd) * Re) * Ri - (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Ri) * Ro) ] ])


class VoltageDivider(Circuit):
    def __init__(self, fs: int, R1_val: float, R2_val: float) -> None:
        self.fs = fs

        self.R1 = Resistor(R1_val)
        self.R2 = Resistor(R2_val)

        self.P1 = ParallelAdaptor(self.R1, self.R2)
        self.I1 = PolarityInverter(self.P1)
        self.Vs = IdealVoltageSource(self.I1)

        elements = [
            self.R1,
            self.R2,
            self.P1,
            self.I1,
            self.Vs,
        ]

        super().__init__(elements, self.Vs, self.Vs, self.R1)

    def set_R1(self,new_R):
        self.R1.set_resistance(new_R)

