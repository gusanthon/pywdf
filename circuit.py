from wdf import *
from rtype import *
import scipy.signal
from scipy.io import wavfile
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


    def plot_freqz(self, delta_dur: float = 1, amp: float = 1):
        x = self.get_impulse_response(delta_dur = delta_dur, amp = amp)
        w, h = scipy.signal.freqz(x, 1, 2**15)

        magnitude = 20 * np.log10(np.abs(h) + np.finfo(float).eps)
        phase = np.angle(h)
        magnitude_peak = np.max(magnitude)
        top_offset = 10
        bottom_offset = 70
        frequencies = w / (2 * np.pi) * self.fs

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

        plt.show()


    def __impedance_calc(self, R: RTypeAdaptor):
        pass

