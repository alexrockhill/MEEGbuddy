import sys
import os, glob
from mne import find_events
from mne.io import read_raw_brainvision,RawArray,Raw
from mne.channels import read_dig_montage
import numpy as np
from mne import create_info
from tqdm import tqdm


def bv2fif(dataf,corf,new_sfreq=1000.0,use_event_ch=False,event_ch='STI TMS',
           ch_order=None,threshold=0.2,ch_name='Fz',padding=5,min_event_dist=1.8,
           max_event_dist=2.5,eogs=('VEOG','HEOG'),ecg='ECG',preload='default'):

    if preload == 'default':
        preload = os.getcwd() + '/workfile'
    raw = read_raw_brainvision(dataf,eog=eogs,preload=preload)

    if new_sfreq < raw.info['sfreq']:
        if use_event_ch:
            events = find_events(raw,stim_channel=event_ch,output='onset',
                                 min_duration=1/raw.info['sfreq'])
            events = events[:,0]
        else:
            ch = abs(raw._data[raw.ch_names.index(ch_name)])
            sfreq = raw.info['sfreq']
            events = list()
            # find a bunch of events, not all of which will be right
            for i in tqdm(range(100+int(padding*sfreq),len(ch)-100,
                                int(min_event_dist*sfreq))):
                current = list(ch[i:i+int(min_event_dist*sfreq)+1])
                event = max(current)
                j = current.index(event)
                if (event > threshold):
                    events.append(i+j)
                del current
            events = [events[i] for i in range(len(events)) if
                      i == 0 or
                      (abs(events[i] - events[i-1]) > sfreq*min_event_dist and
                       abs(events[i] - events[i-1]) < sfreq*max_event_dist)]

            print('Events found: %i' %(len(events)))

        if int(raw.info['sfreq'])%int(new_sfreq) != 0:
            print('Sampling frequencies are not integer multiples, will not work.')
        else:
            factor = int(raw.info['sfreq'])/int(new_sfreq)

        resampled_events = []
        for i in tqdm(range(len(raw.times)/factor)):
            round_down = range(i*factor,i*factor+factor/2)
            round_up = range(i*factor+factor/2,(i+1)*factor)
            for j in round_down:
                if j in events:
                    resampled_events.append(i)
            for j in round_up:
                if j in events:
                    resampled_events.append(i+1)
        print('Resampled events: %i' %(len(resampled_events)))

        raw_resampled_data = np.zeros((len(raw.ch_names),
                                       len(raw.times)/factor))
        for i,ch_name in enumerate(raw.ch_names):
            print(ch_name)
            ch = raw._data[i]
            raw_resampled_data[i] = ch[::factor]
            del ch

        raw.info['sfreq'] = new_sfreq
        raw_resampled = RawArray(raw_resampled_data,raw.info)

        info = create_info([event_ch], raw_resampled.info['sfreq'],
                           ['stim'],verbose=False)
        arr = np.zeros((1, len(raw_resampled.times)))
        for i in resampled_events:
            arr[0,i:i+100] = 1
        ch = RawArray(arr,info,verbose=False)
        raw_resampled.add_channels([ch],force_update_info=True)
        raw = raw_resampled
    elif new_sfreq < raw.info['sfreq']:
        raise ValueError('Up sampling not written')

    info = create_info(['Cz'], raw.info['sfreq'], ['eeg'],verbose=False)
    Cz = RawArray(np.zeros((1, len(raw.times))), info,verbose=False)
    raw.add_channels([Cz], force_update_info=True)

    raw.set_eeg_reference(['TP9', 'TP10'])
    raw.drop_channels(['TP9', 'TP10'])

    if ch_order is None:
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

    for ch in eogs:
        ch_ix = raw.ch_names.index(ch)
        raw._data[ch_ix, :] *= 1e-6
        ch_order.append(ch)

    ch_ix = raw.ch_names.index(ecg)
    raw._data[ch_ix, :] *= 1e-6
    ch_order.append(ecg)
    raw.set_channel_types({ecg:'ecg'})

    event_ch = [ch for ch in raw.info['ch_names'] if 'STI' in ch][0]
    if event_ch:
        ch_order.append(event_ch)

    raw.pick_channels(ch_order)

    if corf is not None:
        montage = read_dig_montage(bvct=corf)
        raw.set_montage(montage,verbose=False)

    fname = (os.getcwd() + '/' +
             os.path.splitext(os.path.basename(dataf))[0] + '-raw.fif')
    print('Saving to ' + fname)
    raw.save(fname,verbose=False,overwrite=True)
    return raw

if __name__  == '__main__':
    if len(sys.argv) == 2:
        _,dataf = sys.argv
        corf = None
        print('Using only data file, the fif file will not have electrode positions.')
    elif len(sys.argv) != 3:
        raise ValueError('Please provide the .vhdr and the .bvct files')
    else:
        _,dataf,corf = sys.argv
    raw = bv2fif(dataf,corf)
