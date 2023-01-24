import numpy as np
from typing import Callable
from .wdf import baseWDF, rootWDF
from .rtype import RTypeAdaptor
from scipy.io import wavfile
import scipy.signal
import matplotlib.pyplot as plt
from scipy.fftpack import fft


class Circuit:

    def __init__(self, source: baseWDF, root: rootWDF, output: baseWDF) -> None:
        """Initialize Circuit class functionality.

        Args:
            source (baseWDF): the circuit's voltage source
            root (rootWDF): the root of the wdf connection tree
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
        """Get circuit's impulse response 

        Args:
            delta_dur (float, optional): duration of Dirac delta function in seconds. Defaults to 1.
            amp (float, optional): amplitude of delta signal's first sample. Defaults to 1.

        Returns:
            np.array: impulse response of the system
        """
        d = np.zeros(int(delta_dur * self.fs))
        d[0] = amp
        return self.process_signal(d)

    def set_sample_rate(self, new_fs: float) -> None:
        """Change system's sample rate

        Args:
            new_fs (float): sample rate to change circuit to
        """
        if self.fs != new_fs:
            self.fs = new_fs
            for key in self.__dict__:
                if hasattr(self.__dict__[key], 'fs') and self.__dict__[key].fs != new_fs:
                    self.__dict__[key].prepare(new_fs)

    def reset(self) -> None:
        """Return values of each circuit element's incident & reflected waves to 0
        """
        for key in self.__dict__:
            if isinstance(self.__dict__[key], baseWDF):
                self.__dict__[key].reset()

    def plot_freqz(self, outpath: str = None):
        """Plot the circuit's frequency response

        Args:
            outpath (str, optional): filepath to save figure. Defaults to None.
        """
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

    def plot_freqz_list(self, values: list, set_function: Callable, param_label: str = 'value', outpath: str = None):
        """Plot circuit's frequency response(s) while varying a parameter

        Args:
            values (list): list of parameter values to iterate through
            set_function (Callable): circuit's function for setting parameter
            param_label (str, optional): name of parameter being modulated. Defaults to 'value'.
            outpath (str, optional): filepath to save figure. Defaults to None.
        """
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
            print(f'{param_label} : {value}')
            set_function(value)
            x = self.get_impulse_response()

            h = fft(x, fft_size)[:N2]
            magnitude = 20 * np.log10(np.abs(h) + np.finfo(float).eps)
            phase = np.angle(h)
            magnitude_peak = np.max(magnitude)
            top_offset = 10
            bottom_offset = 70

            xlims = [10**0, 10 ** np.log10(self.fs / 2)]
            ax[0].semilogx(frequencies, magnitude, label = f'{param_label} : {value}')
            ax[0].set_xlim(xlims)
            ax[0].set_ylim([magnitude_peak - bottom_offset, magnitude_peak + top_offset])
            ax[0].set_xlabel("Frequency [Hz]")
            ax[0].set_ylabel("Magnitude [dBFs]")
            ax[0].set_title(loc = 'left', label = self.__class__.__name__ + ' magnitude response')
            ax[0].grid(True)
            ax[0].legend()

            phase = 180 * phase / np.pi
            ax[1].semilogx(frequencies, phase, label = f'{param_label} : {value}')
            ax[1].set_xlim(xlims)
            ax[1].set_ylim([-180, 180])
            ax[1].set_xlabel("Frequency [Hz]")
            ax[1].set_ylabel("Phase [degrees]")
            ax[1].set_title(loc = 'left', label = self.__class__.__name__ + ' phase response')
            ax[1].grid(True)
            ax[1].legend()

        plt.tight_layout()
        if outpath:
            plt.savefig(outpath)        
        
        plt.show()


    def AC_transient_analysis(self, freq: float = 1000, amplitude: float = 1, t_ms: float = 1, outpath: str = None):
        """Plot transient analysis of Circuit's response to sine wave

        Args:
            freq (float, optional): frequency of sine wave. Defaults to 1000.
            amplitude (float, optional): amplitude of sine wave. Defaults to 1.
            t_ms (float, optional): time in ms of sine wave. Defaults to 5.
        """
        _, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 6.5))

        n_samples = int(t_ms * self.fs / 100)

        n = np.arange(0, 2, 1/self.fs) 
        x = np.sin(2 * np.pi * freq * n) * amplitude
        y = self.process_signal(x)

        ax[0].plot(x[:n_samples], label="input signal")
        ax[0].plot(y[:n_samples], label="output signal")
        ax[0].set_xlabel("sample")
        ax[0].set_ylabel("amplitude")
        ax[0].set_title(loc = 'left', label = "output signal vs input signal waveforms")

        ax[0].legend()

        N = len(n)
        w = scipy.signal.windows.hann((len(n)))
        y_fft = scipy.fft.fft(w * y)
        yf = scipy.fft.fftfreq(N, 1 / self.fs)[20 : N // 2]

        ax[1].plot(yf, 2.0 / N * np.abs(y_fft[20 : N // 2]))
        ax[1].set_xlabel("frequency [Hz]")
        ax[1].set_ylabel("magnitude")
        ax[1].set_title(loc = 'left', label = "output signal spectrum")
        ax[1].grid(True)

        if outpath:
            plt.savefig(outpath)

        plt.show()
        

    def _impedance_calc(self, R: RTypeAdaptor):
        """Placeholder function used to calculate impedance of Rtype adaptor

        Args:
            R (RTypeAdaptor): Rtype of which to calculate impedance
        """
        pass
