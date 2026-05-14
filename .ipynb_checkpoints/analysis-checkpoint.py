import matplotlib.pyplot as plt
import numpy as np
import sys
import src.time_series as time_series
import src.acoustic_entrainment as acoustic_entrainment
from src.time_series import CollectionTDMS as ctdms
from src.acoustic_entrainment import mic_response

def mic_correct(c, taps = 151, lfs = 0.68e-3):
    """
    mic_correct uses the impulse response function of the microphone
    to correct the signal using a digital filter (scipy.signal.filtfilt).
    :param c: collection object of TDMS object.
    :param lfs: lfs of microphone (provided by manufacturer).
    :return: list of times of the collection data and 
             list of corrected signal of the collection data.
    """
    import acoustic_entrainment
    import scipy.signal as sig
    import numpy as np
    # creating array of gain values using acoustic_entrainment.dBs_orig
    gains = lfs * 10 ** (acoustic_entrainment.dBs_orig / 20)
    # making the filter using sig.firwin2
    # 1st value provided is the number of taps for the filter
    # 2nd value is the range of frequencies for the filter -- specified in acoustic_entrainment
    # 3rd value is the array of gains created above
    # 4th value is the maximum frequency times two
    # this filter is also converted into a minimum phase filter (sig.minimum_phase)
    # lastly, the filter is translated into its inverse using ifft and 1 / fft
    filt = np.real(np.fft.ifft(1 / np.fft.fft(sig.minimum_phase(sig.firwin2(taps, np.r_[0, acoustic_entrainment.fs_orig], np.r_[0, gains], fs = 2 * acoustic_entrainment.fs_orig[-1])))))
    return sig.filtfilt(filt, [1], c.x)

def calc_cal_factor(l_col, m_col, deviation, m_mn = 3.5e-4, m_mx = 5e-4):
    """
    
    calc_cal_factor calculates the calibration factor for an individual
    shot in a collection array. Calibration is to the microphone data
    which we have a pretty accurate mV/Pa conversion.
    :param l_col: laser collection object for a single shot.
    :param m_col: microphone collection object for a single shot.
    :param deviation: the number of points to each side of the trough 
                      that the calibration is to include in its calculation.
    :return: calibration factor for a shot.
    
    """
    def find_nearest(array, value):
        return (np.abs(np.asarray(array) - value)).argmin()
    m_trough = np.nonzero(m_col.x == min(m_col.time_gate(tmin = m_mn, tmax = m_mx)[1]))[0][0]
    l_trough = find_nearest(l_col.t, m_col.t[m_trough])
    if deviation != 0:
        return np.mean(m_col.x[m_trough - deviation : m_trough + deviation] / l_col.x[l_trough - deviation : l_trough + deviation])
    return m_col.x[m_trough] / l_col.x[l_trough]

def calc_calibration_factors(laser, mic, dat_i):
    cals = []
    for col_i in range(1, len(laser.get_data()[dat_i].collection)):
        t = mic_tau_shift(laser, mic, dat_i, col_i)
        mic.get_data()[dat_i].apply("shift", tau = t, inplace = True)
        cals.append(calc_cal_factor(laser.get_data()[dat_i].collection[col_i], mic.get_data()[dat_i].collection[col_i], 0))
        mic.get_data()[dat_i].apply("shift", tau = -t, inplace = True)
    return cals

def mic_tau_shift(s1, s2, dat_i, col_i, m_mn = 3.5e-4, m_mx = 5e-4, l_mn = 4.4e-4, l_mx = 4.6e-4) -> float: 
    # assuming s1 will always be laser and s2 will always be microphone
    # also assume that the data will be preprocessed at this point:
    # - the data must already be detrended
    # - the data doesn't have to be lowpassed or bin_averaged, but it can be beforehand
    try:
        x0 = []
        for s in [s1, s2]:
            c = s.get_data()[dat_i].collection[col_i]
            if s.get_name().startswith("mic"):
                mn = c.t[np.nonzero(c.x == np.min(c.time_gate(tmin = m_mn, tmax = m_mx)[1]))[0][0]]
                mx = c.t[np.nonzero(c.x == np.max(c.time_gate(tmin = m_mn, tmax = mn)[1]))[0][0]]
            else:
                mn = c.t[np.nonzero(c.x == np.min(c.time_gate(tmin = l_mn, tmax = l_mx)[1]))[0][0]]
                mx = c.t[np.nonzero(c.x == np.max(c.time_gate(tmin = l_mn, tmax = mn)[1]))[0][0]]
            x = [c.time_gate(tmin = mx, tmax = mn)[0][np.nonzero(np.diff(np.sign(c.time_gate(tmin = mx, tmax = mn)[1])))[0][0]], c.time_gate(tmin = mx, tmax = mn)[0][np.nonzero(np.diff(np.sign(c.time_gate(tmin = mx, tmax = mn)[1])))[0][0] + 1]]
            y = [c.x[np.nonzero(c.t == x[0])[0][0]], c.x[np.nonzero(c.t == x[1])[0][0]]]
            x0.append(x[0] - y[0] * ((x[1] - x[0]) / (y[1] - y[0])))
        return x0[1] - x0[0]
    except IndexError:
        print("Tau failed at index", col_i, "in data set #", str(dat_i) + "!")
        return 0

def graph_systems(systems = [], title = "", save = False) -> None:
    import matplotlib.pyplot as plt
    plt.rcParams["figure.figsize"] = (54, 15)
    fig, ax = plt.subplots(2, 9)
    for s in systems:
        ax = ax.flatten()
        fig.suptitle("setups vs frequency (strong shot and phi 82-160)")
        for i in range(len(s.get_snr_vs_freq()[0])):
            if i == 0:
                ax[i].set_title("Strong shot")
                ax[i].plot(s.get_snr_freq_range(), s.get_snr_vs_freq()[:, i], label = s.get_name())
                ax[i].set_ylabel("snr")
                ax[i].set_xlabel("frequency")
            else:
                ax[i].set_title("Phi " + str(s.get_phis()[i - 1]))
                ax[i].plot(s.get_snr_freq_range(), s.get_snr_vs_freq()[:, i], label = s.get_name())
                ax[i].set_ylabel("snr")
                ax[i].set_xlabel("frequency")
    fig.tight_layout(pad = 1.5)
    ax[0].legend()
    if save:
        fig.savefig(title)
    plt.show()
    return None

def phi(p):
    from scipy.special import erfinv
    return np.sqrt(2) * erfinv(2 * p - 1)

def expected_max(n):
    mun = phi(1 - 1 / n)
    sigman = phi(1 - 1 / (n * np.e)) - mun
    return mun + sigman * 0.577
    

def std_max(n, mu):
    mun = phi(1 - 1 / n) + mu
    sigman = phi(1 - 1 / (n * np.e)) - mun
    return sigman * np.pi * np.sqrt(1/6)
    

class System():

    def __init__(self, name = "", data_files = [], power = 19, snr_freq_cut = 0, phis = [], snr_resolution = 10, snr_freq_range = [1e4, 2e6], channel = "", set_data = False) -> None:
        self.set_name(name)
        self.set_power(power)
        self.set_phis(phis)
        self.set_snr_resolution(snr_resolution)
        self.set_snr_freq_cutoff(snr_freq_cut)
        self.set_snr_freq_range(snr_freq_range)
        self.set_channel(channel)
        self.set_df(np.array(data_files))
        if set_data:
            self.set_data(self.get_df())
            return
        self.set_data()
        return

    def get_name(self) -> str:
        """
        
        get_name gets the name of the system.
        :return: the name of the system.
        
        """
        return self.__name

    def set_name(self, n) -> None:
        """
        
        set_name sets the name of the system
        :param n: name of the system.
        :return: None.
        
        """
        self.__name = n
        return None

    def set_channel(self, ch) -> None:
        """
        
        set_channel sets the name of the channel in the ctdms object.
        :param ch: channel name, if provided, should be "X" or "Y".
        :return: None.
        
        """
        self.__channel = ch
        return None

    def get_channel(self) -> str:
        """

        get_channel gets the name of the channel that was set for
        the ctdms object or returns a blank str.
        :return: name of the channel.
        
        """
        return self.__channel
    
    def get_df(self) -> np.typing.NDArray:
        """
        
        get_df gets the list of data files that the system has.
        :return: the list of data files in the system
        
        """
        return self.__df

    def set_df(self, df) -> None:
        """
        
        set_df sets the list of data files.
        :param df: list of data files.
        :return: None.
        
        """
        self.__df = df
        return None

    def get_data(self) -> np.typing.NDArray:
        """
        
        get_data gets the data collections for the system.
        :return: list of data collections.
        
        """
        return self.__data

    def set_data(self, df = [], ind = -1, mic_correct = False, tmin = None, tmax = None) -> None:
        """
        
        set_data sets the data collections for every data file provided for the system.
        :param df: list of data files provided in the system object.
        :return: None.
        
        """
        if len(df) != 0:
            self.__data = np.empty((len(df), ), dtype=object)
            for d in range(len(df)):
                self.__data[d] = ctdms(df[d])
                if self.get_channel() == "X":
                    self.__data[d].set_collection(self.get_channel(), tmin = tmin, tmax = tmax)
                elif self.get_channel() == "Y":
                    self.__data[d].set_collection(self.get_channel(), tmin = tmin, tmax = tmax)
                else:
                    if self.get_name().startswith("mic"):
                        self.__data[d].set_collection("Y", tmin = tmin, tmax = tmax)
                        if mic_correct:
                            self.__data[d].apply("correct", response = mic_response, recollect = True)
                    else:
                        self.__data[d].set_collection("X", tmin = tmin, tmax = tmax)
                        if self.get_name()[:] == "sagnac":
                            self.__data[d].apply("calibrate", cal = -1, inplace = True)
        elif len(df) == 0 and ind == -1:
            self.__data = np.empty((len(self.get_df()), ), dtype=object)
        else:
            self.__data[ind] = ctdms(self.get_df()[ind])
            if self.get_channel() == "X":
                self.__data[ind].set_collection(self.get_channel(), tmin = tmin, tmax = tmax)
            elif self.get_channel() == "Y":
                self.__data[ind].set_collection(self.get_channel(), tmin = tmin, tmax = tmax)
            else:
                if self.get_name().startswith("mic"):
                    self.__data[ind].set_collection("Y", tmin = tmin, tmax = tmax)
                    if mic_correct:
                        self.__data[ind].apply("correct", response = mic_response, recollect = True)
                else:
                    self.__data[ind].set_collection("X", tmin = tmin, tmax = tmax)
                    if self.get_name()[:] == "sagnac":
                        self.__data[ind].apply("calibrate", cal = -1, inplace = True)
        return None


    def get_power(self) -> int:
        """
        
        get_power gets the power of the pulsed laser.
        :return: integer value of the laser power.
        
        """
        return self.__power

    def set_power(self, p) -> None:
        """
        
        set_power sets the power of the pulsed laser used in the system.
        :param p: power of the pulsed laser.
        :return: None.
        
        """
        self.__power = p
        return None

    def get_snr_freq_cutoff(self) -> float:
        """
        
        get_snr_freq_cutoffs gets the frequency cutoff.
        :return: tuple containing the starting and ending frequency of the cutoffs.
        
        """
        return self.__snr_freq_cutoff

    def set_snr_freq_cutoff(self, freq) -> None:
        """
        
        set_snr_freq_cutoffs sets the frequency cutoffs range.
        :freqs: frequency cutoff that the snr will be run at.
        :return: None.
        
        """
        self.__snr_freq_cutoff = freq
        return None

    def get_phis(self) -> list:
        """
        
        get_phis gets the phis that the system scans over in the minimum detectable setup.
        :return: None.
        
        """
        return self.__phis

    def set_phis(self, p) -> None:
        """
        
        set_phis sets the phis that the system scans over to an array.
        :param p: list of phis.
        :return: None.
        
        """
        self.__phis = p
        return None

    # add comments for these functions down!
    def set_snr_resolution(self, res) -> None:
        """"""
        self.__snr_res = res
        return None

    def get_snr_resolution(self) -> int:
        """"""
        return self.__snr_res
        
    def set_snr_freq_range(self, ran) -> None:
        """"""
        self.__snr_freq_range = ran
        return None
    def get_snr_freq_range(self) -> list:
        """"""
        return self.__snr_freq_range
    
    def calc_snr_of_collection_at_cutoff(self, ind, col, f, signal = (0, 0), noise = (0, 0), bins = False) -> float:
        self.__data[ind].apply("detrend", mode = "constant", inplace = True)
        self.__data[ind].apply("lowpass", cutoff = f, inplace = True)
        if bins:
            self.__data[ind].apply("bin_average", Npts = int(self.__data[ind].r / (2 * f)), inplace = True)
        peak = np.max(np.abs(self.__data[ind].collection[col].time_gate(tmin = signal[0], tmax = signal[1])[1]))
        rms = (expected_max(len(self.__data[ind].collection[col].time_gate(tmin = noise[0], tmax = noise[1])[0])) * np.std(self.__data[ind].collection[col].time_gate(tmin = noise[0], tmax = noise[1])[1]))
        self.set_data(ind = ind)
        return  peak / rms

    def calc_snr_at_cutoff(self, i, f, signal = (0, 0), noise = (0, 0), bins = False) -> float:
        """"""
        self.__data[i].apply("detrend", mode = "constant", inplace = True)
        self.__data[i].apply("lowpass", cutoff = f, inplace = True)
        if bins:
            self.__data[i].apply("bin_average", Npts = int(self.__data[i].r / (2 * f)), inplace = True)
        peaks = np.array([])
        rms = np.array([])
        for s in self.__data[i].collection:
            peaks = np.append(peaks, np.max(np.abs(s.time_gate(tmin = signal[0], tmax = signal[1])[1])))
            rms = np.append(rms, np.std(s.time_gate(tmin = noise[0], tmax = noise[1])[1]))
        snr = np.mean(peaks / (rms * expected_max(len(s.time_gate(tmin = noise[0], tmax = noise[1])[0]))))
        self.set_data(ind = i)
        return snr
    
    def set_snr_vs_freq(self, snr = np.array([]), i = -1) -> None:
        """"""
        if len(snr) == 0:
            self.__snr_vs_freq = [np.array([], dtype = float) for x in range(len(self.get_data()))]
            return None
        elif i != -1:
            self.__snr_vs_freq[i] = np.append(self.__snr_vs_freq[i], snr)
            return None
        self.__snr_vs_freq = snr
        return None
    
    def reset_snr_vs_freq(self) -> None:
        """"""
        self.__snr_vs_freq = [np.array([], dtype = float) for x in range(len(self.get_data()))]
        return None

    def calc_snr_vs_freq(self, j = -1, freq = [0, 0], signal = (0, 0), noise = (0, 0), bins = False) -> np.typing.NDArray:
        """"""
        snr = np.array([])
        if freq[0] == 0 and freq[1] == 0:
            freq = np.linspace(*self.get_snr_freq_range(), self.get_snr_resolution())
        else:
            freq = np.linspace(freq[0], freq[1], self.get_snr_resolution())
        if j != -1:
            for i in range(len(freq)):
                snr = np.append(snr, self.calc_snr_at_cutoff(j, freq[i], signal, noise, bins))
            return snr
        else:
            for k in range(len(self.get_data())):
                for i in range(len(freq)):
                    snr = np.append(snr, self.calc_snr_at_cutoff(k, freq[i], signal, noise, bins))
            return snr
    
    def get_snr_vs_freq(self) -> np.typing.NDArray:
        """"""
        return self.__snr_vs_freq
