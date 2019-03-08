import os.path as op
from mne.io import Raw
from scipy.io import savemat

def _fname(mb,process_dir,keyword,ftype,*tags):
    # must give process dir, any tags
    if process_dir == 'plots':
        fname = mb.process_dirs[process_dir]
    else:
        fname = op.join(mb.process_dirs[process_dir],mb.subject)
    if mb.task:
        fname += '_' + str(mb.task)
    if mb.eeg:
        fname += '_eeg'
    if mb.meg:
        fname += '_meg'
    for tag in tags:
        if tag:
            fname += '_' + str(tag)
    fname += '-' + keyword
    if ftype:
        fname += '.' + str(ftype)
    return fname


def _has_raw(mb,keyword=None):
    return op.isfile(_fname(mb,'raw','raw','fif',keyword))


def _load_raw(mb,keyword=None):
    if op.isfile(_fname(mb,'raw','raw','fif',keyword)):
        raw = Raw(_fname(mb,'raw','raw','fif',keyword),
                              verbose=False,preload=True)
        print('%s' %(keyword)*(keyword is not None)+' raw data loaded.')
    elif keyword is None:
        f = mb.fdata[0]
        print(f)
        raw = Raw(f, preload=True, verbose=False)
        raw.info['bads'] = []
        for f in mb.fdata[1:]:
            print(f)
            r = Raw(f, preload=True, verbose=False)
            r.info['bads'] = []
            raw.append(r)
        if mb.eeg:
            raw = raw.set_eeg_reference(ref_channels=[],projection=False)
        raw = raw.pick_types(meg=mb.meg,eeg=mb.eeg,stim=True,
                             eog=True,ecg=True,emg=True)
    else:
        raise ValueError('No raw data file found ' +
                         'for %s' %(keyword)*(keyword is not None))
    return raw


def _save_raw(mb,raw,keyword=None):
    print('Saving raw' + ' %s' %(keyword)*(keyword is not None))
    raw.save(_fname(mb,'raw','raw','fif',keyword),
             verbose=False,overwrite=True)


def _has_ICA(mb,event=None,keyword=None):
    if event is None:
        return op.isfile(_fname(mb,'raw','ica','fif',keyword))
    else:
        return op.isfile(_fname(mb,'epochs','ica','fif',keyword,event))

def _load_ICA(mb,event=None,keyword=None):
    if event is None:
        fname = _fname(mb,'raw','ica','fif',keyword)
    else:
        fname = _fname(mb,'epochs','ica','fif',keyword,event)
    if op.isfile(fname):
        ica = read_ica(fname)
        print('ICA loaded.')
        return ica
    else:
        print('No ICA data file found %s'
                %(keyword if keyword is not None else ''))


def _save_ICA(mb,ica,event=None,keyword=None):
    print('Saving ICA %s' %(keyword if keyword is not None else ''))
    if event is None:
        ica.save(_fname(mb,'raw','ica','fif',keyword))
    else:
        ica.save(_fname(mb,'epochs','ica','fif',keyword,event))


def _has_epochs(mb,event,keyword=None):
    return op.isfile(_fname(mb,'epochs','epo','fif',event,keyword))


def _load_epochs(mb,event,keyword=None):
    if not _has_epochs(mb,event,keyword=None):
        raise ValueError(event + ' epochs must be made first' +
                         ' for %s' %(keyword)*(keyword is not None))
    epochs = read_epochs(_fname(mb,'epochs','epo','fif',event,keyword),
                         verbose=False,preload=True)
    print('%s epochs loaded' %(event) +
          ' for %s' %(keyword)*(keyword is not None))
    epochs._data = epochs._data.astype('float64') # mne bug work-around
    return epochs


def _save_epochs(mb,epochs,event,keyword=None):
    print('Saving epochs for ' + event +
          ' %s' %(keyword)*(keyword is not None))
    epochs.save(_fname(mb,'epochs','epo','fif',event,keyword))


def _has_evoked(mb,event,keyword=None):
    return op.isfile(_fname(mb,'epochs','ave','fif',event,keyword))


def _load_evoked(mb,event,keyword=None):
    if not _has_evoked(mb,event,keyword=keyword):
        raise ValueError(event + ' evoked must be made first' +
                         ' for %s' %(keyword)*(keyword is not None))
    evoked = read_evokeds(_fname(mb,'epochs','ave','fif',event,keyword),
                          verbose=False)
    print('%s epochs loaded' %(event) +
          ' for %s' %(keyword)*(keyword is not None))
    return evoked[0]


def _save_evoked(mb,evoked,event,keyword=None):
    print('Saving evoked for ' + event +
          ' %s' %(keyword)*(keyword is not None))
    evoked.save(_fname(mb,'epochs','ave','fif',event,keyword))


def _has_autoreject(mb,event):
    return op.isfile(_fname(mb,'epochs','ar','npz',event))


def _load_autoreject(mb,event):
    if _has_autoreject(mb,event):
        f = np.load(_fname(mb,'epochs','ar','npz',event))
        return f['ar'].item(),f['reject_log'].item()
    else:
        print('Autoreject must be run for ' + event)

def _save_autoreject(mb,event,ar,reject_log):
    np.savez_compressed(_fname(mb,'epochs','ar','npz',event),ar=ar,
                        reject_log=reject_log)


def _has_TFR(mb,event,condition,value,keyword=None):
    fname = _fname(mb,'analyses','tfr','npy',event,condition,value,keyword)
    fname1b = _fname(mb,'analyses','tfr_params','npz',event,condition,value,
                         keyword)
    fname2 = _fname(mb,'TFR','tfr','npz',event,condition,value,keyword)
    return ((op.isfile(fname) and op.isfile(fname1b)) or
             op.isfile(fname2))


def _load_TFR(mb,event,condition,value,keyword=None):
   fname = _fname(mb,'analyses','tfr','npy',event,condition,value,keyword)
   fname1b = _fname(mb,'analyses','tfr_params','npz',event,condition,value,
                         keyword)
   fname2 = _fname(mb,'TFR','tfr','npz',event,condition,value,keyword)
   if op.isfile(fname) and op.isfile(fname1b):
       tfr = np.load(fname)
       f = np.load(fname1b)
       frequencies,n_cycles = f['frequencies'],f['n_cycles']
   elif op.isfile(fname2):
       f = np.load(fname2)
       tfr,frequencies,n_cycles = f['tfr'],f['frequencies'],f['n_cycles']
   else:
       raise ValueError('No TFR to load for %s %s %s'
                        %(event,condition,value))
   print('TFR loaded for %s %s %s' %(event,condition,value))
   return tfr,frequencies,n_cycles


def _save_TFR(mb,tfr,frequencies,n_cycles,
             event,condition,value,keyword,compressed=True):
   print('Saving TFR for %s %s %s' %(event,condition,value))
   if compressed:
       np.savez_compressed(_fname(mb,'analyses','tfr','npz',
                                       event,condition,value,keyword),
                           tfr=tfr,frequencies=frequencies,
                           n_cycles=n_cycles)
   else:
       np.save(_fname(mb,'analyses','tfr','npy',
                           event,condition,value,keyword),tfr)
       np.savez_compressed(_fname(mb,'analyses','tfr_params','npz',
                                       event,condition,value,keyword),
                           frequencies=frequencies,n_cycles=n_cycles)


def _CPT_decider(mb,event,condition,value,tfr=False,band=None):
    if band:
        fname = _fname(mb,'analyses','CPT','npz',event,condition,value,band)
    elif tfr:
        fname = _fname(mb,'analyses','CPT','npz',event,condition,value,'tfr')
    else:
        fname = _fname(mb,'analyses','CPT','npz',event,condition,value)
    return fname


def _has_CPT(mb,event,condition,value,tfr=False,band=None):
    return op.isfile(_CPT_decider(mb,event,condition,value,tfr=tfr,
                                            band=band))


def _load_CPT(mb,event,condition,value,tfr=False,band=None):
    if _has_CPT(mb,event,condition,value,tfr=tfr,band=band):
        f = np.load(_CPT_decider(mb,event,condition,value,tfr=tfr,
                                      band=band))
        print('Cluster permuation test loaded for %s %s %s'
              %(event,condition,value))
        if band:
            return f['clusters'],f['cluster_p_values'],f['band']
        elif tfr:
            return f['clusters'],f['cluster_p_values'],f['frequencies']
        else:
            return f['clusters'],f['cluster_p_values']
    else:
        raise ValueError('Cluster permuation test not found for %s %s %s'
                         %(event,condition,value))


def _save_CPT(mb,event,condition,value,clusters,cluster_p_values,
              times,frequencies=None,band=None):
    print('Saving CPT for %s %s %s' %(event,condition,value))
    if band:
        np.savez_compressed(_fname(mb,'analyses','CPT','npz',
                                   event,condition,value,band),
                            clusters=clusters,
                            cluster_p_values=cluster_p_values,band=band)
    elif frequencies:
        np.savez_compressed(_fname(mb,'analyses','CPT','npz',
                                   event,condition,value,'tfr'),
                            clusters=clusters,frequencies=frequencies,
                            cluster_p_values=cluster_p_values)
    else:
        np.savez_compressed(_fname(mb,'analyses','CPT','npz',
                                   event,condition,value),
                            clusters=clusters,
                            cluster_p_values=cluster_p_values)

def _has_inverse(mb,event,condition,value,keyword=None):
    fname = _fname(mb,'source_estimates','inv','fif',keyword,
                        event,condition,value)
    fname2 = _fname(mb,'source_estimates','inverse_params','npz',
                         keyword,event,condition,value)
    return op.isfile(fname) and op.isfile(fname2)


def _load_inverse(mb,event,condition,value,keyword=None):
    if mb._has_inverse(event,condition,value,keyword=keyword):
        fname = _fname(mb,'source_estimates','inv','fif',keyword,
                        event,condition,value)
        fname2 = _fname(mb,'source_estimates','inverse_params','npz',
                             keyword,event,condition,value)
        f = np.load(fname2)
        return (read_inverse_operator(fname),f['lambda2'].item(),
                f['method'].item(),f['pick_ori'].item())
    else:
        print('Inverse not found for %s %s %s' %(event,condition,value))


def _save_inverse(mb,inv,lambda2,method,pick_ori,
                  event,condition,value,keyword=None):
    print('Saving inverse for %s %s %s' %(event,condition,value))
    write_inverse_operator(_fname(mb,'source_estimates','inv','fif',
                                       keyword,event,condition,value),
                           inv,verbose=False)
    np.savez_compressed(_fname(mb,'source_estimates','inverse_params','npz',
                                    keyword,event,condition,value),
                        lambda2=lambda2,method=method,pick_ori=pick_ori)


def _has_source(mb,event,condition,value,keyword=None,fs_av=False):
    fname = _fname(mb,'sources','source-lh','.stc',keyword,
                        event,condition,value,'fs_av'*fs_av)
    return op.isfile(fname)


def _load_source(mb,event,condition,value,keyword=None,fs_av=False):
    fname = _fname(mb,'sources','source-lh','.stc',keyword,
                        event,condition,value,'fs_av'*fs_av)
    if op.isfile(fname):
        print('Fs average s'*fs_av + 'S'*(not fs_av) + 'ource loaded for '+
              '%s %s %s' %(event,condition,value))
        return read_source_estimate(fname)
    else:
        print('Source not found for %s %s %s' %(event,condition,value))


def _save_source(mb,stc,event,condition,value,keyword=None,fs_av=False):
    if fs_av:
        print('Saving source fs average for %s %s %s' %(event,condition,
                                                        value))
        stc.save(_fname(mb,'source_estimates','source',None,keyword,
                             event,condition,value,'fs_av'),ftype='stc')
    else:
        print('Saving source for %s %s %s' %(event,condition,value))
        stc.save(_fname(mb,'source_estimates','source',None,keyword,
                             event,condition,value),ftype='stc')


def _has_PSD_image(mb,keyword,ch,N,deltaN,fmin,fmax,NW):
    fname = _fname(mb,'analyses','image','npz',ch,
                        'N_%i_dN_%.2f' %(N,deltaN),
                        'fmin_%.2f_fmax_%.2f_NW_%i' %(fmin,fmax,NW))
    return op.isfile(fname)

def _load_PSD_image(mb,keyword,ch,N,deltaN,fmin,fmax,NW):
    if mb._has_PSD_image(keyword,ch,N,deltaN,fmin,fmax,NW):
        fname = _fname(mb,'analyses','image','npz',ch,
                            'N_%i_dN_%.2f' %(N,deltaN),
                            'fmin_%.2f_fmax_%.2f_NW_%i' %(fmin,fmax,NW))
        print('Loading image')
        return np.load(fname)['image']
    else:
        return None

def _save_PSD_image(mb,image,keyword,ch,N,deltaN,fmin,fmax,NW):
    print('Saving psd multitaper image')
    np.savez_compressed(_fname(mb,'analyses','image','npz',ch,
                        'N_%i_dN_%.2f' %(N,deltaN),
                        'fmin_%.2f_fmax_%.2f_NW_%i' %(fmin,fmax,NW)),
                        image=image)


def remove(mb,event=None,keyword=None):
    dir_name = 'raw' if event is None else 'epochs'
    suffix = 'epo' if event else 'raw'
    fname = _fname(mb,dir_name,suffix,'fif',event,keyword)
    if op.isfile(fname):
        os.remove(fname)


def raw2mat(mb,keyword=None,ch=None):
    raw = mb._load_raw(keyword=keyword)
    if ch is None:
        ch_dict = mb._get_ch_dict(raw)
    else:
        ch_dict = {raw.info['ch_names'].index(ch):ch}
    raw_data = raw.get_data()
    data_dict = {}
    for ch in ch_dict:
        data_dict[ch_dict[ch]] = raw_data[ch]
    savemat(_fname(mb,'raw','data','mat',keyword),data_dict)