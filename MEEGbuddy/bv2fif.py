import sys
import os, glob
import os.path as op
from mne import find_events, Epochs, EpochsArray
from mne.io import read_raw_brainvision,RawArray
from mne.channels import read_dig_montage
import numpy as np
from mne import create_info,events_from_annotations
from tqdm import tqdm

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

    ch = raw._data[raw.info['ch_names'].index('FC5')].copy()
    b,a = signal.butter(3,1./(raw.info['sfreq']/2),'highpass')
    ch = signal.filtfilt(b,a,ch)
    fig,ax = plt.subplots()
    minx = int(raw.info['sfreq']*10)
    maxx = int(raw.info['sfreq']*40)
    ax.plot(np.arange(minx,maxx)/raw.info['sfreq'],ch[minx:maxx])
    fig.show()

    min_event_dist = 1.5 #float(raw_input('Minimum Event Distance?    '))
    max_event_dist = 4 #float(raw_input('Maximum Event Distance?    '))

    done = False
    while not done:
        threshold = float(raw_input('Threshold?    '))

        step = int(raw.info['sfreq']*min_event_dist)
        # find a bunch of events, not all of which will be right
        print('Finding events')
        events = list()
        for i in tqdm(range(step, len(ch)-step, 2*step)):
            max_index = np.argmax(abs(ch[i-step:i+step]))
            dist = np.sort(abs(ch[i-step:i+step]))
            compare_value = dist[-2]
            if ch[i-step+max_index] - compare_value > threshold:
                events.append(i - step + max_index)
        ok = False
        i = 0
        indices = np.arange(len(events))
        np.random.shuffle(indices)
        while not ok and i < len(events):
            fig,ax = plt.subplots()
            ax.plot(ch[int(events[indices[i]]-raw.info['sfreq']):int(events[indices[i]]+raw.info['sfreq'])])
            fig.show()
            i += 1
            ok = input('Enter to keep testing, type anything to stop')
        done = input('Enter to reset threshold, type anything to finish')

    # make a channel
    info = create_info(['DBS'], raw.info['sfreq'],['stim'],verbose=False)
    arr = np.zeros((1, len(raw.times)))
    for i in events:
        arr[0,i:i+100] = 1
    event_ch = RawArray(arr,info,verbose=False)
    return event_ch

def bv2fif(dataf,corf,ch_order=None,eogs=('VEOG','HEOG'),ecg='ECG',emg='EMG',
           preload='default',ref_ch='Fp1',dbs=False,new_sfreq=1000.0):
    montage = read_dig_montage(bvct=corf)
    if preload == 'default':
        preload = os.path.dirname(dataf)+'/workfile'

    raw = read_raw_brainvision(dataf,preload=preload)

    if dbs:
        event_ch = get_events(raw)

    # save downsampled raw for multitaper spectrogram
    raw_data = np.zeros((raw._data.shape[0],
                         int(raw._data.shape[1]/raw.info['sfreq']*new_sfreq)))
    raw_info = raw.info.copy()
    raw_info['sfreq'] = new_sfreq
    for i in tqdm(range(len(raw._data))):
        ch = raw._data[i,::int(raw.info['sfreq']/new_sfreq)]
        raw_data[i] = ch
        del ch

    raw_resampled = RawArray(raw_data,raw_info)
    raw_resampled.set_annotations(raw.annotations)

    if dbs:
        old_event_ch = [ch for ch in raw.info['ch_names'] if 'STI' in ch]
        if old_event_ch:
            raw_resampled.drop_channels([old_event_ch[0]])
        event_ch._data = event_ch._data[:,::int(raw.info['sfreq']/new_sfreq)]
        event_ch.info['sfreq'] = new_sfreq
        event_ch.__len__ = len(event_ch._data[0])
        event_ch.info['lowpass'] = raw_resampled.info['lowpass']
        raw_resampled.add_channels([event_ch])

    prepInst(raw_resampled,dataf,'raw',montage,ref_ch,eogs,ecg,emg)

    events,event_ids = events_from_annotations(raw)
    if len(np.unique(events[:,2])) > 1:
        events = events[np.where(events[:,2] == events[1,2])[0]] #skip new segment
    epochs = Epochs(raw,events,tmin=-2,tmax=2,proj=False,
                    preload=op.dirname(dataf)+'/workfile-epo',
                    baseline=(-0.5,-0.1),verbose=False,detrend=1)
    events = events[epochs.selection] #in case any epochs don't have data and get thrown out (poorly placed at beginning or end)
    epo_data = np.zeros((epochs._data.shape[0],epochs._data.shape[1],
                         int(np.ceil(epochs._data.shape[2]/epochs.info['sfreq']*new_sfreq))))
    for i in tqdm(range(epochs._data.shape[0])):
        for j in range(epochs._data.shape[1]):
            epo_curr = epochs._data[i,j,::int(epochs.info['sfreq']/new_sfreq)]
            epo_data[i,j] = epo_curr
            del epo_curr
    events[:,0] = np.array(events[:,0]*new_sfreq/raw.info['sfreq'],dtype=int)
    epo_resampled = EpochsArray(epo_data,epochs.info.copy(),events,tmin=-2)
    epo_resampled.info['sfreq'] = new_sfreq
    epo_resampled.events[:,2] = np.arange(len(events))
    epo_resampled.event_id = {str(i):i for i in range(len(events))}

    prepInst(epo_resampled,dataf,'epo',montage,ref_ch,eogs,ecg,emg)


def prepInst(inst,dataf,suffix,montage,ref_ch,eogs,ecg,emg):
    info = create_info([ref_ch], inst.info['sfreq'], ['eeg'],verbose=False)
    info['lowpass'] = inst.info['lowpass']
    if suffix == 'raw':
        ref = RawArray(np.zeros((1, len(inst.times))),info,verbose=False)
    elif suffix == 'epo':
        ref = EpochsArray(np.zeros((len(inst),1,len(inst.times))),info,verbose=False)
    inst = inst.add_channels([ref]) #, force_update_info=True)
    #
    inst = inst.set_eeg_reference(['TP9', 'TP10'])
    if suffix == 'epo':
        while len(inst.picks) != len(inst): # weird picks bug
            inst.picks = np.append(inst.picks,len(inst.picks))
    inst = inst.drop_channels(['TP9', 'TP10'])
    #
    ch_order = ['Fp1', 'Fp2', 'AFp1', 'AFp2', 'AF7', 'AF3', 'AFz',
               'AF4', 'AF8', 'AFF5h', 'AFF6h', 'F7', 'F5', 'F3', 'F1',
               'Fz', 'F2', 'F4', 'F6', 'F8', 'FFT9h', 'FFT7h', 'FFC3h',
               'FFC4h', 'FFT8h', 'FFT10h', 'FT9', 'FT7', 'FC5', 'FC3',
               'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10',
               'FTT9h', 'FCC5h', 'FCC1h', 'FCC2h', 'FCC6h', 'FTT10h',
               'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8',
               'TTP7h', 'CCP3h', 'CCP4h', 'TTP8h', 'TP7', 'CP5', 'CP3',
               'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TPP9h',
               'CPP5h', 'CPP1h', 'CPP2h', 'CPP6h', 'TPP10h', 'P7',
               'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
               'PPO5h', 'PPO6h', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
               'POO9h', 'O1', 'POO1', 'Oz', 'POO2', 'O2', 'POO10h']
    #
    for ch in eogs:
        ch_ix = inst.ch_names.index(ch)
        inst._data[ch_ix, :] *= 1e-6
        ch_order.append(ch)
        inst.set_channel_types({ch:'eog'})
    #
    ch_ix = inst.ch_names.index(ecg)
    inst._data[ch_ix, :] *= 1e-6
    ch_order.append(ecg)
    inst.set_channel_types({ecg:'ecg'})
    #
    ch_ix = inst.ch_names.index(emg)
    inst._data[ch_ix, :] *= 1e-6
    ch_order.append(emg)
    inst.set_channel_types({emg:'emg'})
    #
    inst = inst.set_montage(montage,verbose=False)
    #
    inst = inst.reorder_channels(ch_order)
    #
    fname = (os.path.join(os.path.dirname(dataf),
             os.path.basename(dataf).split('.')[0]+'-%s.fif' %(suffix)))
    print('Saving to ' + fname)
    if suffix == 'raw':
        inst.save(fname,verbose=False,overwrite=True)
    else:
        inst.save(fname,verbose=False)

if __name__  == '__main__':
    if len(sys.argv) != 3:
        raise ValueError('Please provide the .vhdr and the .bvct files')

    _,dataf,corf = sys.argv
    bv2fif(dataf,corf)
