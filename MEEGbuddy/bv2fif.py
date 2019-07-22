import sys
import os, glob
import os.path as op
from mne import find_events, Epochs, EpochsArray
from mne.io import read_raw_brainvision, RawArray
from mne.channels import read_dig_montage
import numpy as np
from mne import create_info, events_from_annotations
from tqdm import tqdm

ch_name_order = \
    ['Left', 'Right', 'Nasion', 'Fp1', 'Fpz', 'Fp2', 
     'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFz',
     'AF2', 'AF4', 'AF6', 'AF8', 'AF10',
     'F9', 'F7', 'F5', 'F3', 'F1', 'Fz',
     'F2', 'F4', 'F6', 'F8', 'F10',
     'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz',
     'FC2', 'FC4', 'FC6', 'FT8', 'FT10',
     'T9', 'T7', 'C5', 'C3', 'C1', 'Cz',
     'C2', 'C4', 'C6', 'T8', 'T10',
     'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz',
     'CP2', 'CP4', 'CP6', 'TP8', 'TP10',
     'P9', 'P7', 'P5', 'P3', 'P1', 'Pz',
     'P2', 'P4', 'P6', 'P8', 'P10',
     'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POz',
     'PO2', 'PO4', 'PO6', 'PO8', 'PO10', 'O1', 'Oz',
     'O2', 'I1', 'Iz', 'I2',
     'AFp9h', 'AFp7h', 'AFp5h', 'AFp3h', 'AFp1h',
     'AFp2h', 'AFp4h', 'AFp6h', 'AFp8h', 'AFp10h',
     'AFF9h', 'AFF7h', 'AFF5h', 'AFF3h', 'AFF1h',
     'AFF2h', 'AFF4h', 'AFF6h', 'AFF8h', 'AFF10h',
     'FFT9h', 'FFT7h', 'FFC5h', 'FFC3h', 'FFC1h',
     'FFC2h', 'FFC4h', 'FFC6h', 'FFT8h', 'FFT10h',
     'FTT9h', 'FTT7h', 'FCC5h', 'FCC3h', 'FCC1h',
     'FCC2h', 'FCC4h', 'FCC6h', 'FTT8h', 'FTT10h',
     'TTP9h', 'TTP7h', 'CCP5h', 'CCP3h', 'CCP1h',
     'CCP2h', 'CCP4h', 'CCP6h', 'TTP8h', 'TTP10h',
     'TPP9h', 'TPP7h', 'CPP5h', 'CPP3h', 'CPP1h',
     'CPP2h', 'CPP4h', 'CPP6h', 'TPP8h', 'TPP10h',
     'PPO9h', 'PPO7h', 'PPO5h', 'PPO3h', 'PPO1h',
     'PPO2h', 'PPO4h', 'PPO6h', 'PPO8h', 'PPO10h',
     'POO9h', 'POO7h', 'POO5h', 'POO3h', 'POO1h',
     'POO2h', 'POO4h', 'POO6h', 'POO8h', 'POO10h',
     'OI1h', 'OI2h', 'Fp1h', 'Fp2h',
     'AF9h', 'AF7h', 'AF5h', 'AF3h', 'AF1h',
     'AF2h', 'AF4h', 'AF6h', 'AF8h', 'AF10h',
     'F9h', 'F7h', 'F5h', 'F3h', 'F1h',
     'F2h', 'F4h', 'F6h', 'F8h', 'F10h',
     'FT9h', 'FT7h', 'FC5h', 'FC3h', 'FC1h',
     'FC2h', 'FC4h', 'FC6h', 'FT8h', 'FT10h',
     'T9h', 'T7h', 'C5h', 'C3h', 'C1h',
     'C2h', 'C4h', 'C6h', 'T8h', 'T10h',
     'TP9h', 'TP7h', 'CP5h', 'CP3h', 'CP1h',
     'CP2h', 'CP4h', 'CP6h', 'TP8h', 'TP10h',
     'P9h', 'P7h', 'P5h', 'P3h', 'P1h',
     'P2h', 'P4h', 'P6h', 'P8h', 'P10h',
     'PO9h', 'PO7h', 'PO5h', 'PO3h', 'PO1h',
     'PO2h', 'PO4h', 'PO6h', 'PO8h', 'PO10h',
     'O1h', 'O2h', 'I1h', 'I2h',
     'AFp9', 'AFp7', 'AFp5', 'AFp3', 'AFp1', 'AFpz',
     'AFp2', 'AFp4', 'AFp6', 'AFp8', 'AFp10',
     'AFF9', 'AFF7', 'AFF5', 'AFF3', 'AFF1', 'AFFz',
     'AFF2', 'AFF4', 'AFF6', 'AFF8', 'AFF10',
     'FFT9', 'FFT7', 'FFC5', 'FFC3', 'FFC1', 'FFCz',
     'FFC2', 'FFC4', 'FFC6', 'FFT8', 'FFT10',
     'FTT9', 'FTT7', 'FCC5', 'FCC3', 'FCC1', 'FCCz',
     'FCC2', 'FCC4', 'FCC6', 'FTT8', 'FTT10',
     'TTP9', 'TTP7', 'CCP5', 'CCP3', 'CCP1', 'CCPz',
     'CCP2', 'CCP4', 'CCP6', 'TTP8', 'TTP10',
     'TPP9', 'TPP7', 'CPP5', 'CPP3', 'CPP1', 'CPPz',
     'CPP2', 'CPP4', 'CPP6', 'TPP8', 'TPP10',
     'PPO9', 'PPO7', 'PPO5', 'PPO3', 'PPO1', 'PPOz',
     'PPO2', 'PPO4', 'PPO6', 'PPO8', 'PPO10',
     'POO9', 'POO7', 'POO5', 'POO3', 'POO1', 'POOz',
     'POO2', 'POO4', 'POO6', 'POO8', 'POO10',
     'OI1', 'OIz', 'OI2', 'T3', 'T5', 'T4', 'T6', 'Centroid']  # from chlocs_ALLCH.mat kindly provided by Massimini group

def get_events(raw):
    ''' with the DBS, events are not able to be triggered so we have to use
    the pulses to determine the events'''
    from mne.io import RawArray, Raw
    import numpy as np
    from mne import create_info,Epochs,make_fixed_length_events,pick_types,find_events
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import glob,re
    from scipy import interpolate
    from scipy import signal
    #
    ch = raw._data[raw.info['ch_names'].index(raw.ch_names[3])].copy()
    b,a = signal.butter(3,0.5,'highpass')
    ch = signal.filtfilt(b,a,ch)
    #
    min_event_dist = 1.5 #float(input('Minimum Event Distance?    '))
    max_event_dist = 4 #float(input('Maximum Event Distance?    '))
    #
    done = False
    while not done:
        fig,ax = plt.subplots()
        minx = int(raw.info['sfreq']*10)
        maxx = int(raw.info['sfreq']*40)
        ax.plot(np.arange(minx,maxx)/raw.info['sfreq'],ch[minx:maxx])
        plt.show()
        threshold = None
        while not threshold:
            try:
                threshold = float(input('Threshold?    '))
            except:
                threshold = None
        step = int(raw.info['sfreq']*min_event_dist)
        # find a bunch of events, not all of which will be right
        print('Finding events')
        events = list()
        for i in tqdm(range(step, len(ch)-step, 2*step)):
            max_index = np.argmax(abs(ch[i-step:i+step]))
            dist = np.sort(abs(ch[i-step:i+step]))
            compare_value = dist[-10]
            if ch[i-step+max_index] - compare_value > threshold:
                events.append(i - step + max_index)
        ok = False
        i = 0
        indices = np.arange(len(events))
        np.random.shuffle(indices)
        while not ok and i < len(events):
            fig,ax = plt.subplots()
            ax.plot(ch[int(events[indices[i]]-raw.info['sfreq']):int(events[indices[i]]+raw.info['sfreq'])])
            plt.show()
            i += 1
            ok = input('Enter to keep testing, type anything to stop\n')
        done = input('%i events found. Enter to reset threshold, type anything to finish\n' %(len(events)))
    #
    # make a channel
    info = create_info(['DBS'], raw.info['sfreq'],['stim'],verbose=False)
    arr = np.zeros((1, len(raw.times)))
    for i in events:
        arr[0,i:i+100] = 1
    event_ch = RawArray(arr,info,verbose=False)
    return event_ch

def bv2fif(dataf, corf, ch_order=None, aux=('VEOG', 'HEOG', 'ECG', 'EMG'),
           preload='default', ref_ch='Fp1', dbs=False,
           use_find_events='dbs', tmin=-2, tmax=2, baseline=(-0.5,-0.1),
           detrend=1):
    """Function to convert .eeg, .vmrk and .vhdr BrainVision files to a 
       combined .fif format file.
    Parameters
    ----------
    dataf : str
        The .vhdr file that contains references to the data
    corf : str
        The COR.fif file that contains the montage for this
        particular subject. This will be generated through
        mne_analyze, possibly while using the TMS-EEG GUI.
    ch_order : list of str | None
        If not 'None', a custom channel order is used.
    aux : tuple 
        Auxillary/accessory electrodes to be included in the data.
    preload : 'default' | False
        If false, load data into RAM, if true memory map to disk
    ref_ch : str
        Reference channel used
    dbs: bool
        If true stim channels are named 'DBS' and use_find_events
        is true by default.
    use_find_events : bool | 'dbs'
        If true or 'dbs' and dbs is true, then the peak amplitude
        will be used to find stimulus markers.
    tmin: float
        Time when epochs start
    tmax: float
        Time when epochs end
    baseline: tuple (float, float) | None
        Time to use for baseline mean subtraction
    detrend: int
        1 for linear detrend, 0 for mean subtraction and None for nothing
    Notes
    -----
    An older version of MNE contained a bug that prevented adding 
    channels while using memory mapping. A script to circumvent 
    this also exists.
    """
    use_find_events = ((dbs and use_find_events == 'dbs') or 
                       (isinstance(use_find_events, bool) and 
                        use_find_events))
    if preload == 'default':
        preload = os.path.dirname(dataf) + '/workfile'
    #
    raw = read_raw_brainvision(dataf, preload=preload)
    if corf is None:
        montage = None
    elif '.bvct' in op.basename(corf):
        montage = read_dig_montage(bvct=corf)
    elif '.csv' in op.basename(corf):
        montage = read_dig_montage(csv=corf)
    else:
        raise ValueError('corf not understood')
    #
    if ch_order is None:
        if all([ch in ch_name_order for ch in raw.ch_names]):
            order_dict = {ch: ch_name_order.index(ch) for ch in raw.ch_names}
            ch_order = sorted(order_dict, key=order_dict.get)
        else:  # if no channel order is given and we get names we didn't expect, just sort the channels
            ch_order = sorted(inst.ch_names)
    #
    if use_find_events:
        event_ch = get_events(raw)
        old_event_ch = [ch for ch in raw.info['ch_names'] if 'STI' in ch]
        if old_event_ch:
            raw.drop_channels([old_event_ch[0]])
        raw.add_channels([event_ch])
        if use_find_events and use_find_events != 'dbs':
            raw.rename_channels({'DBS':'TMS'})
    else:
        events, event_ids = events_from_annotations(raw)
    #
    prepInst(raw, dataf, 'raw', montage, ref_ch, aux,
             'DBS' if dbs else 'TMS', ch_order)
    #
    if len(np.unique(events[:,2])) > 1:
        events = events[np.where(events[:,2] == events[-1, 2])[0]] #skip new segment
    epochs = Epochs(raw, events, tmin=tmin, tmax=tmax, proj=False,
                    preload=preload, baseline=baseline, verbose=False,
                    detrend=detrend)
    events = events[epochs.selection]  #in case any epochs don't have data and get thrown out (poorly placed at beginning or end)
    epochs.event_id = {str(i):i for i in range(len(events))}
    #
    prepInst(epochs, dataf, 'epo', montage, ref_ch, aux,
             'DBS' if dbs else 'TMS', ch_order)
    if isinstance(preload, str) and op.isfile(preload):
        os.remove(preload)


def prepInst(inst, dataf, suffix, montage, ref_ch, aux, stim, ch_order):
    if ref_ch is not None:
        info = create_info([ref_ch], inst.info['sfreq'], ['eeg'], verbose=False)
        info['lowpass'] = inst.info['lowpass']
        if suffix == 'raw':
            ref = RawArray(np.zeros((1, len(inst.times))),info,verbose=False)
        elif suffix == 'epo':
            ref = EpochsArray(np.zeros((len(inst), 1, len(inst.times))),info,verbose=False)
        inst = inst.add_channels([ref]) #, force_update_info=True)
    #
    inst = inst.set_eeg_reference(ref_channels='average', projection=False,
                                  verbose=False)
    if suffix == 'epo':
        while len(inst.picks) != len(inst.ch_names): # weird picks bug
            inst.picks = np.append(inst.picks, len(inst.picks))
    #
    if aux is not None:
        for ch in aux:
            try:
                ch_ix = inst.ch_names.index(ch)
                if suffix == 'raw':
                    inst._data[ch_ix] *= 1e-6
                elif suffix == 'epo':
                    inst._data[:, ch_ix] *= 1e-6
                ch_order.append(ch)
                inst.set_channel_types({ch: 'eog'})
            except Exception as e: 
                print(e, '%s channel not working' % ch)
    #
    if suffix != 'epo':
        if stim in inst.ch_names:
            ch_ix = inst.ch_names.index(stim)
            ch_order.append(stim)
            inst.set_channel_types({stim: 'stim'})
    #
    if montage is not None:
        inst = inst.set_montage(montage, verbose=False)
    #
    inst = inst.reorder_channels(ch_order)
    #
    fname = (os.path.join(os.path.dirname(dataf),
             os.path.basename(dataf).split('.')[0]+'-%s.fif' % suffix))
    print('Saving to ' + fname)
    if suffix == 'raw':
        inst.save(fname, verbose=False, overwrite=True)
    else:
        inst.save(fname, verbose=False)

if __name__  == '__main__':
    if len(sys.argv) != 3:
        raise ValueError('Please provide the .vhdr and the .bvct or .csv files')

    _, dataf, corf = sys.argv
    bv2fif(dataf,corf)
