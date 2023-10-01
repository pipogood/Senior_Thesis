from pylsl import StreamInlet, resolve_stream
import numpy as np
from scipy.fft import fft2, fftshift
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from scipy.signal import welch
import mne

# first resolve an EEG stream on the lab network
print("looking for an EEG stream...")
streams = resolve_stream()
print(streams)

data = []
time_count = 0.0

# create a new inlet to read from the stream
inlet = StreamInlet(streams[0])

def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

fs = 250.0
lowcut = 8.0
highcut = 30.0
 
channels = ['C3','Cz','C4']

info = mne.create_info(
    ch_names= channels,
    ch_types= ['eeg']*len(channels),
    sfreq= fs
)

#size 3,246

while True:
    # get a new sample (you can also omit the timestamp part if you're not
    # interested in it)
    sample, timestamp = inlet.pull_sample()
    data.append(sample[1:4])

    if timestamp:

        time_count += 1 * 0.004
        time_count = np.round(time_count ,3)
        data2 = np.asarray(data)
        data2 = np.transpose(data)
        # data2 = butter_bandpass_filter(data2, lowcut, highcut, fs, order=6)

        if time_count == 2:
            raw = mne.io.RawArray(data2, info)
            filter = raw.copy().filter(l_freq=8.0, h_freq=30.0, method = 'iir', iir_params= {"order": 6, "ftype":'butter'})
            psd:np.ndarray = filter.compute_psd(method='welch', fmax = 30).get_data()
            # frequencies, psd = welch(data2, fs=60, nperseg=250)
            # print(type(frequencies))

            # plt.figure(figsize=(8, 4))
            # plt.semilogy(frequencies, np.transpose(psd))
            # plt.title('Power Spectral Density (PSD)')
            # plt.xlabel('Frequency (Hz)')
            # plt.ylabel('PSD (V^2/Hz)')    
            # plt.grid(True)
            # plt.show()
            # plt.show()
            print(timestamp, data2, np.shape(data2))
            print(".................................................")
            print(filter.get_data(), np.shape(filter.get_data()))
            print(".................................................")
            print(psd, np.shape(psd))
            break
