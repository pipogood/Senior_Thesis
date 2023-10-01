"""
🚀Deploy utilities.py v1.0 (for use in Online task)

contains :
- read_xdf() - epoching() - decoding()
"""
import pyxdf
from pyxdf import match_streaminfos, resolve_streams

import mne
from mnelab.io.xdf import read_raw_xdf

import matplotlib.pyplot as plt
from pprint import pprint
from sklearn import metrics

from mne.decoding import Scaler


def read_xdf(filename: str, bandpass=(None, 45.0), show_plot=True, show_psd=True, verbose=False, plot_scale=169) -> mne.io.array.array.RawArray:
    """
    Loading XDF file into MNE-RawArray. MNE-Python does not support this file format out of the box, 
    but we can use the pyxdf package and MNELAB to import the data. 

    attribute:
        bandpass  : set Bandpass filter (l_freq, h_freq)
        show_plot : If True, show all EEG channels and able to zoom in-out, scaling
        show_psd  : If True, show overall average power spectral density

    return: MNE RawArray
    """
    # Read xdf
    # https://github.com/cbrnr/bci_event_2021#erders-analysis-with-mne-python
    streams = resolve_streams(filename) # streams has 2 types: EEG, Markers
    if verbose: pprint(streams) 

    # Find stearms type EEG
    stream_id = match_streaminfos(streams, [{"type": "EEG"}])[0]
    raw = read_raw_xdf(filename, stream_ids=[stream_id])
    # print(raw.info['bads'])

    # Set channel types
    #   set_channel_types({CHANNEL_NAME : CHANNEL_TYPE}) 
    raw.set_channel_types({'obci_eeg1_0': 'eeg'})   # FP1
    raw.set_channel_types({'obci_eeg1_1': 'eeg'})   # O1
    raw.set_channel_types({'obci_eeg1_2': 'eeg'})   # Oz
    raw.set_channel_types({'obci_eeg1_3': 'eeg'})   # O2
    raw.set_channel_types({'obci_eeg1_4': 'eeg'})   # POz
    raw.set_channel_types({'obci_eeg1_5': 'eeg'})   # Pz
    raw.set_channel_types({'obci_eeg1_6': 'eeg'})   # none
    raw.set_channel_types({'obci_eeg1_7': 'eeg'})   # none

    # Add Bandpass filtering (default 0Hz - 45Hz)
    raw = raw.filter(l_freq=bandpass[0], h_freq=bandpass[1])
    
    show = False
    # Plot EEG graph
    # https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.plot
    if show_plot:

        raw.plot(
            duration=15, 
            start=0, 
            scalings=plot_scale, # You may edit scalings value later
        ) #, n_channels=8, bad_color='red'
        show = True

    # https://mne.tools/stable/generated/mne.io.RawArray.html#mne.io.RawArray.compute_psd
    if show_psd:
        raw.compute_psd(
            fmax=60,
            picks=[
                'obci_eeg1_1',
                'obci_eeg1_2',
                'obci_eeg1_3',
                'obci_eeg1_4',
                'obci_eeg1_5',
            ],                      # pick by channel name
            # picks='eeg',          # pick by channel type
            ).plot()
        show = True
        
    if show: plt.show()

    return raw


def epoching(raw: mne.io.array.array.RawArray, tmin=0, tmax=10, baseline=(0,0), filename=None, show_eeg=False, show_psd=True, show_time_freq=False, plot_scale=200) -> mne.epochs.Epochs:
    """
    Epoching, showing Power spectral density (PSD) plot, split by Left-Right stimuli event, average by epoch 

    attribute:
        tmin            : <int or float> Initial timestamp of epoch (if is 0 means trigger timestamp same as event start)
        tmax            : <int or float> Final timestamp (if is 10 means set epoch duration 10 second)
        show_eeg        : If True, (same as show_plot) show all EEG channels and able to zoom in-out, scaling
        show_psd        : If True, show power spectral density split by Left-Right stimuli
        show_time_freq  : If True, show Time-Frequency plot split by Left-Right stimuli and each O1, Oz, O2, POz, Pz

    return: MNE Epochs
    """
    raw_eeg = raw.pick_channels([
                    'obci_eeg1_1',
                    'obci_eeg1_2',
                    'obci_eeg1_3',
                    'obci_eeg1_4',
                    'obci_eeg1_5',
                ])

    events, event_dict = mne.events_from_annotations(raw_eeg)

    epochs = mne.Epochs(raw_eeg, events, 
        tmin=tmin,      # init timestamp of epoch (if is 0 means trigger timestamp same as event start)
        tmax=tmax,      # final timestamp (if is 10 means set epoch duration 10 second)
        baseline=baseline,
        preload=True,
    )

    # Visualization
    show = False

    if show_eeg:
        # raw_eeg.plot(
        #     duration=10, 
        #     start=0, 
        #     scalings=plot_scale, # You may edit scalings value later
        # ) #, n_channels=8, bad_color='red'

        epochs['2'].plot(
            scalings=plot_scale, # You may edit scalings value later
            title='Left stimuli start',
        )
        epochs['5'].plot(
            scalings=plot_scale, # You may edit scalings value later
            title='Right stimuli start',
        )
        show = True

    # Plot Power spectral density
    if show_psd:
        fig, ax = plt.subplots(2, figsize=(9,9))

        epochs['2'].compute_psd(
            fmax=30,                    
            # method='welch',
            ).plot(
                axes=ax[0],
                average=True, 
                )
        ax[0].set_title('Left stimuli' if not filename else 'Left stimuli - '+filename)

        epochs['5'].compute_psd(
            fmax=30,                    
            # method='welch',
            ).plot(
                axes=ax[1],
                average=True, 
                )
        ax[1].set_title('Right stimuli' if not filename else 'Right stimuli - '+filename)
        plt.tight_layout()
        show = True

    # Plot Time-frequency
    if show_time_freq:
        # Split Epochs (Trials) including Cue
        epochs_from_cue = mne.Epochs(raw_eeg, events, 
                tmin=-1.0,      # init timestamp of epoch (-1.0 means trigger before event start 1.0 second)
                tmax=10.0,      # final timestamp (10 means set epoch duration 10 second)
                # baseline=(0, 0),
                preload=True,
            )
        channel_name = (    
            '(O1) obci_eeg1_1',
            '(Oz) obci_eeg1_2',
            '(O2) obci_eeg1_3',
            '(POz) obci_eeg1_4',
            '(Pz) obci_eeg1_5',)

        fig, ax = plt.subplots(2, 5, figsize=(17, 7))
        plt.title('Time-frequency')

        # Select range of frequency you wanna plot
        show_freqs = list(range(1,25))
        # show_freqs = [6.0, 12.0, 18.0, 24.0, 10.0, 20.0, 30.0]

        for i in range(5):
            power_L = mne.time_frequency.tfr_multitaper(
                epochs_from_cue['2'], 
                freqs=show_freqs, 
                n_cycles=10, 
                use_fft=True, 
                decim=3,
                return_itc=False,
            )
            power_L.plot([i], mode='logratio', axes=ax[0,i], show=False, colorbar=False)
            ax[0,i].set_title('Left stimuli - '+channel_name[i]); ax[0,i].set_xlabel('')

            power_R = mne.time_frequency.tfr_multitaper(
                epochs_from_cue['5'], 
                freqs=show_freqs, 
                n_cycles=10, 
                use_fft=True, 
                decim=3,
                return_itc=False,
            )
            power_R.plot([i], mode='logratio', axes=ax[1,i], show=False, colorbar=False)
            ax[1,i].set_title('Right stimuli - '+channel_name[i])
        plt.tight_layout()
        show = True

    if show: plt.show()

    return epochs


def decoding(epochs: mne.epochs.Epochs, plot=False, verbose=False) -> list:
    """
    To Decoding the epochs, convert them into a NumPy array and then feed them into models. 
    Afterward, evaluate the performance of the models and report the results.

    attribute:
        plot    : If True, visualize plot all events, compare the two ranges of frequencies, and view the outputs.
        verbose : If True, print the outputs and classification report in the terminal.

    return: List of output
    """
    outputs = list()

    # Pick only event 2: Left stimuli, 5: Right stimuli
    epochs = epochs['2','5']

    # Get the values as numpy array
    X:np.ndarray = epochs.get_data() * 1e6
    F:np.ndarray = epochs.compute_psd(method='welch', fmax=30).get_data()
    t:np.ndarray = epochs.times
    y:np.ndarray = epochs.events[:, -1]

    # model = simpleModel
    
    test_L, test_R, label, x = [], [], [], list(range(16))

    for evnt in x:
        
        test_L.append(F[evnt][2][6])
        test_R.append(F[evnt][2][10])
        label.append(y[evnt])

    outputs = simpleModel(test_L, test_R)

    if plot:
        fig, ax = plt.subplots(2, figsize=(10, 5))

        ax[0].step(x, test_L)
        ax[0].step(x, test_R)
        ax[0].legend(['frequency 6Hz', 'frequency 10Hz'])

        ax[1].step(x, label)
        ax[1].step(x, outputs)
        ax[1].legend(['Labels', 'Predictions'])

        plt.tight_layout()
        plt.show()

    if verbose:
        print('\nLabels      :', label)
        print('Predictions :', outputs, '\n')

        # Classification performance
        classification_report = metrics.classification_report(label, outputs)
        # confusion_matrix = metrics.confusion_matrix(label, outputs)
        print( classification_report )

        # Sensitivity & Specificity
        # print('Sensitivity :', metrics.recall_score(label, outputs))
        # print('Specificity :', metrics.recall_score(label, outputs, pos_label=?))

    return outputs


if __name__=='__main__':

    filename = 'Pipo_1_5_test1.xdf'
    # filename = 'Pipo_1_5_test2.xdf'
    # filename = 'Pipo_1_5_test3.xdf'

    # Loading XDF file into MNE-RawArray
    raw = read_xdf(filename, 
        bandpass=(3.0, 15.0), # (default 0Hz - 45Hz)

        show_plot=False, 
        # show_plot : If True, show all EEG channels and able to zoom in-out, scaling

        show_psd=False,
        # show_psd : If True, show overall average power spectral density
    )

    # Epoching, showing Power spectral density (PSD) split by Left-Right stimuli event
    epochs = epoching(raw, filename,

        show_eeg=False,
        # show_eeg : If True, show all EEG channels and able to zoom in-out, scaling split by Left-Right stimuli

        show_psd=False,
        # show_psd : If True, show overall average power spectral density split by Left-Right stimuli

        show_time_freq=False,
        # show_time_freq : If True, show Time-Frequency plot split by Left-Right stimuli and each O1, Oz, O2, POz, Pz
    )

    # Decoding
    outputs = decoding(epochs,
        plot=False,
        # plot    : If True, visualize plot all events, compare the two ranges of frequencies, and view the outputs.

        verbose=True,
        # verbose : If True, print the outputs and classification report in the terminal.
    )