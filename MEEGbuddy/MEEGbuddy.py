import os
import os.path as op

try:
    import glob, re, json
    import numpy as np
    from tqdm import tqdm
    from pandas import read_csv, DataFrame
except:
    raise ImportError('Unable to import core tools (pandas, glob, re, json, ' +
                      'tqdm)... must install to continue')
try:
    import matplotlib.pyplot as plt
except:
    raise ImportError('Unable to import matplotlib, install to continue.')


class MEEGbuddy:
    '''
    Takes in one or more .fif files for a subject that are combined together
    into an MNE raw object and one behavior csv file assumed to be combined.
    The bad channels are then auto-marked,
    ICA is then auto-marked and then plots are presented for you to ok,
    epochs are made, and autoreject is applied.
    All data is saved automatically in BIDS-inspired format.
    '''
    def __init__(self, subject=None, session=None, run=None, fdata=None, 
                 behavior=None, behavior_description=None, baseline=None, 
                 stimuli=None, eeg=False, meg=False, ecog=False, seeg=False, 
                 response=None, task=None, no_response=None,
                 exclude_response=None, tbuffer=1, subjects_dir=None,
                 epochs=None, event=None, fs_subjects_dir=None, bemf=None,
                 srcf=None, transf=None, preload=True, seed=551832, file=None):
        '''
        subject : str 
            the name/id of the subject
        session : str
            the name/id of the session
        run : str 
            the name/id of the run
        fdata : str|list of str 
            one or more .fif files with event triggers and EEG data
        behavior : str 
            a csv file with variable names indexing a list of attributes
        behavior_description : dict 
            dictionary desribing the behavior attributes
        baseline : list [channel (str), start (float), stop(float)]|
                       [(channel (str), event_id (int)), start (float), stop(float)]
                  a list of stim channel or stim channel and height of the trigger
                  and start and stop times for the baseline.
        stimuli : dict 
                A dictionary of the names of the stimuli paired with a list of
                channel, start and stop as descibed in baseline
        eeg : bool
            whether eeg data is present
        meg : bool
            whether meg data is present
        ecog : bool
            whether ecog data is present
        seeg : bool
            whether seeg data is present
        response : list
            a channel, start and stop list described in baseline
        no_response : list of int 
            indices of behavior to exclude due to no response
        exclude_response : list of int
            indices of behavior to exclude (no response not necessary)
        t_buffer : int
            amount of time to buffer epochs for time frequency analyses
        subjects_dir : str
            directory where derivative data should be stored
        epochs : mne.Epochs object
            an epochs object to save in order to start with epochs instead of raw data
        event : str
            the name of the event for the epochs if starting with epochs
        fs_subjects_dir : str
            the filepath of the freesurfer reconstruction directory
        bemf : str
            the filepath of the boundary element model file
        srcf : str
            the filepath of the source file
        transf : str

        '''
        from mne import set_log_level
        import warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        set_log_level("ERROR")

        if file is None:
            meta_data = {}

            if subjects_dir is None:
                subjects_dir = os.getcwd()
                print('No subjects_dir supplied defaulting to current working directory')

            meta_data['Subjects Directory'] = subjects_dir

            name = str(subject) + '_'
            name += str(session) + '_' if session is not None else ''
            name += str(run) + '_' if run is not None else ''
            name += str(task) + '_' if task is not None else ''
            for dt, v in {'meg': meg, 'eeg': eeg, 'ecog': ecog, 'seeg': seeg}.items():
                if v:
                    name += dt
            file = op.join(subjects_dir, 'meta_data', name + '.json')
            if not op.isdir(os.path.dirname(file)):
                os.makedirs(os.path.dirname(file))

            if fdata is None:
                raise ValueError('Please supply raw file or list of files to combine')

            if not isinstance(fdata, list):
                fdata = [fdata]

            meta_data['Functional Data'] = fdata

            meta_data['MEG'] = meg
            meta_data['EEG'] = eeg
            meta_data['ECOG'] = ecog
            meta_data['SEEG'] = seeg

            if subject is None:
                raise ValueError('Subject name is required to differentiate subjects')

            meta_data['Subject'] = subject
            meta_data['Session'] = session
            meta_data['Run'] = run
            meta_data['Task'] = task

            processes = ['meta_data', 'raw', 'epochs', 'source_estimates', 'plots',
                         'analyses', 'behavior', 'preprocessing']

            meta_data['Process Directories'] = {}
            for process in processes:
                meta_data['Process Directories'][process] = \
                    op.join(subjects_dir, process)

            if behavior is None:
                behavior = op.join(meta_data['Process Directories']['behavior'], subject)
                if session is not None:
                    behavior = op.join(behavior, session)
                if run is not None:
                    behavior = op.join(behavior, run)
                behavior = op.join(behavior, name + '.csv')
            else:
                try:
                    df = read_csv(behavior)
                except:
                    raise ValueError('Behavior must be the path to a csv file')
                if behavior_description is None:
                    behavior_description = {col: 'Please describe column here' for col in df.columns}
                else:
                    if not all([des in list(df.columns) for des in behavior_description]):
                        raise ValueError('All behavior columns are not described in ' +
                                         'behavior description dictionary')
            
            meta_data['Behavior'] = behavior
            meta_data['Behavior Description'] = behavior_description

            meta_data['No Response'] = \
                [] if no_response is None else no_response

            meta_data['Exclude Response'] = \
                [] if exclude_response is None else exclude_response

            meta_data['Events'] = stimuli
            if any([len(stimuli[event]) != 3 for event in stimuli]):
                raise ValueError('There must be a channel, start and stop time for each stimulus.')

            if response:
                if len(response) != 3:
                    raise ValueError('response must contain a channel, start time and stop time.')
            meta_data['Response'] = response

            if not baseline or len(baseline) != 3:
                print('Baseline must contain a channel, start time and stop time. ' +
                      'Okay to continue, use normalized=False when making epochs')
            
            meta_data['Baseline'] = baseline
            meta_data['Time Buffer'] = tbuffer
            meta_data['Preload'] = preload
            meta_data['Seed'] = seed

            if fs_subjects_dir is None:
                print('Please provide the SUBJECTS_DIR specified to freesurfer ' +
                      'if you want to do source estimation. These files are not ' +
                      'copied over to save space and to preserve the original ' +
                      'file identies for clarity and troubleshooting. Pass ' +
                      'fs_subjects_dir=False to supress this warning')
            else:
                if fs_subjects_dir and not os.path.isdir(fs_subjects_dir):
                    raise ValueError('fs_subjects_dir not a directory')
            meta_data['Freesufer SUBJECTS_DIR'] = fs_subjects_dir

            if bemf is None:
                if not fs_subjects_dir is None:
                    print('Please provide the file for a boundary element model if ' +
                          'you want source estimation, this can be done using a ' +
                          'FLASH or T1 scan using MNE make_flash_bem or ' +
                          'make_watershed_bem respectively')
            meta_data['Boundary Element Model'] = bemf

            if srcf is None:
                if not fs_subjects_dir is None:
                    print('Please provide the file for a source space if ' +
                          'you want source estimation, this can be done using MNE ' +
                          'setup_source_space')
            meta_data['Source Space'] = srcf

            if transf is None:
                if not fs_subjects_dir is None:
                    print('Please provide the file for a coordinate transformation if ' +
                          'you want source estimation, this can be done using MNE ' +
                          'seting the SUBJECTS_DIR and SUBJECT environmental variables ' +
                          'running \'mne_analyze\', loading the subjects surface from ' +
                          'the recon-all files and the digitization data from the raw file ' +
                          'and then manually adjusting until the coordinate frames match. ' +
                          'This can then be saved out as a coordinate transform file.')
            meta_data['Coordinate Transform'] = transf

            with open(file, 'w') as f:
                json.dump(meta_data, f)

        with open(file, 'r') as f:
            meta_data = json.load(f)
            self.subject = meta_data['Subject']
            self.session = meta_data['Session']
            self.run = meta_data['Run']
            self.fdata = meta_data['Functional Data']
            self.behavior = meta_data['Behavior']
            self.behavior_description = meta_data['Behavior Description']
            self.baseline = meta_data['Baseline']
            self.events = meta_data['Events']
            self.response = meta_data['Response']
            self.eeg = meta_data['EEG']
            self.meg = meta_data['MEG']
            self.ecog = meta_data['ECOG']
            self.seeg = meta_data['SEEG']
            self.task = meta_data['Task']
            self.no_response = meta_data['No Response']
            self.exclude_response = meta_data['Exclude Response']
            self.tbuffer = meta_data['Time Buffer']
            self.process_dirs = meta_data['Process Directories']
            self.subjects_dir = meta_data['Subjects Directory']
            self.fs_subjects_dir = meta_data['Freesufer SUBJECTS_DIR']
            self.bemf = meta_data['Boundary Element Model']
            self.srcf = meta_data['Source Space']
            self.transf = meta_data['Coordinate Transform']
            self.preload = meta_data['Preload']
            self.seed = meta_data['Seed']
            self.file = file

        if epochs is not None and event is not None:
            epochs.save(self._fname('epochs', 'epo', 'fif', event))


    def _add_meta_data(self, meta_dict, overwrite=False):
        with open(self.file, 'r') as f:
            meta_data = json.load(f)
        for meta_key in meta_dict:
            if meta_key in meta_data and not overwrite:
                self._overwrite_error('Meta Data', meta_key)
            meta_data[meta_key] = meta_dict[meta_key]
        with open(self.file, 'w') as f:
            json.dump(meta_data, f)


    def _fname(self, process_dir, keyword, ftype, *tags):
        # must give process dir, any tags
        if process_dir == 'plots':
            dirname = op.join(self.process_dirs[process_dir], keyword)
        else:
            dirname = (op.join(self.process_dirs[process_dir], keyword, self.subject) 
                       if process_dir == 'analyses' else
                       op.join(self.process_dirs[process_dir], self.subject))
            if self.session is not None:
                dirname = op.join(dirname, self.session)
            if self.run is not None:
                dirname = op.join(dirname, self.run)
        if not op.isdir(dirname):
            os.makedirs(dirname)
        fname = self.subject
        if self.session:
            fname += '_' + str(self.session)
        if self.run:
            fname += '_' + str(self.run)
        if self.task:
            fname += '_' + str(self.task)
        if self.eeg:
            fname += '_eeg'
        if self.meg:
            fname += '_meg'
        if self.ecog:
            fname += '_ecog'
        if self.seeg:
            fname += '_seeg'
        for tag in tags:
            if tag:
                fname += '_' + str(tag).replace(' ', '_')
        if not process_dir == 'plots':
            fname += '-' + keyword
        if ftype:
            fname += '.' + str(ftype)
        return op.join(dirname, fname)


    def save2BIDS(self, dirname, overwrite=False):
        from mne_bids import write_raw_bids, write_anat
        from pandas import read_csv
        from subprocess import call
        from mne.io import Raw
        from mne import read_trans
        if not op.isdir(dirname):
            os.makedirs(dirname)
        readmef = op.join(dirname, 'README')
        if not op.isfile(readmef):
            with open(readmef, 'w') as f:
                f.write('Welcome to the ____ dataset, published in _____, ' +
                        'funded by _____, please cite ______\n\n' + 
                        'Edit this file to describe your project')
        if isinstance(self.fdata, list):
            raw = self._load_raw()
            self._save_raw(raw, keyword='bids')
            raw = self._load_raw(keyword='bids')
        else:
            raw = self._load_raw()
        raw = Raw(raw.filenames[0], preload=False)
        session = '01' if self.session is None else self.session
        run = '01' if self.run is None else self.run
        if self.meg:
            modality = 'meg'
        elif self.eeg:
            modality = 'eeg'
        elif self.ecog:
            modality = 'ecog'
        elif self.seeg:
            modality = 'seeg'
        else:
            raise ValueError('Modality error')
        bids_basename = 'sub-%s_ses-%s_task-%s_run-%s' % (self.subject, session, self.task, run)
        write_raw_bids(raw, bids_basename, output_path=dirname, overwrite=overwrite)
        with open(op.join(dirname, 'dataset_description.json'), 'r+') as f:
            description = json.load(f)
            f.seek(0)
            if self.task in description:
                if not all([description[self.task]['Stimuli'] == self.events and
                            description[self.task]['Response'] == self.response and
                            description[self.task]['Baseline'] == self.baseline]):
                    raise ValueError('Task triggers mismatch with previous triggers')
            else:
                description[self.task] = {}
                description[self.task]['Stimuli'] = self.events
                description[self.task]['Response'] = self.response
                description[self.task]['Baseline'] = self.baseline
            json.dump(description, f)
        eventsf = op.join(dirname, 'sub-%s' % self.subject, 'ses-%s' % session, 
                          modality, bids_basename + '_events.tsv')
        df = read_csv(eventsf, sep='\t')
        df2 = read_csv(self.behavior)
        df = df.join(df2)
        df.to_csv(eventsf, sep='\t', index=False, na_rep='n/a')
        behavior_descriptionf = op.join(dirname, 'task-%s_events.json' % self.task)
        if op.isfile(behavior_descriptionf):
            os.remove(behavior_descriptionf)
        with open(behavior_descriptionf, 'w') as f:
            json.dump(self.behavior_description, f)
        if self.fs_subjects_dir:
            t1f = op.join(self.fs_subjects_dir, self.subject, 'mri', 'T1.mgz')
            '''write_anat(dirname, self.subject, t1f, session=session,
                       raw=raw, trans=read_trans(self.transf), overwrite=overwrite)
            '''
            source_fs_and_mne(self.fs_subjects_dir)
            anatdir = op.join(dirname, 'sub-%s' % self.subject, 'anat')
            if not op.isdir(anatdir):
                os.makedirs(anatdir)
            if 'FREESURFER_HOME' not in os.environ:
                print('Freesurfer not sourced, please do this so T1 can be converted and transfered')
            else:
                t1_conf = op.join(anatdir, 'sub-%s_T1w.nii.gz' % self.subject)
                if not op.isfile(t1_conf):
                    call(['mri_convert %s %s' % (t1f, t1_conf)], shell=True, env=os.environ)


    def _has_raw(self, keyword=None):
        return op.isfile(self._fname('raw', 'raw', 'fif', keyword))


    def _load_raw(self, keyword=None):
        from mne.io import Raw
        if keyword is None:
            preload = self.preload if self.preload else op.join(self.subjects_dir,
                                                                'workfile')
            f = self.fdata[0]
            print('Loading Raw file(s)')
            print(f)
            raw = Raw(f, preload=preload, verbose=False)
            raw.info['bads'] = []
            for f in self.fdata[1:]:
                print(f)
                r = Raw(f, preload=preload, verbose=False)
                r.info['bads'] = []
                raw.append(r)
            if self.eeg:
                raw = raw.set_eeg_reference(ref_channels=[], projection=False)
            raw = raw.pick_types(meg=self.meg, eeg=self.eeg, stim=True,
                                 eog=True, ecg=True, emg=True)
        elif op.isfile(self._fname('raw', 'raw', 'fif', keyword)):
            raw = Raw(self._fname('raw', 'raw', 'fif', keyword),
                                  verbose=False, preload=True)
            self._file_loaded('Raw', keyword=keyword)
        else:
            self._no_file_error('Raw', keyword=keyword)
        return self._exclude_unpicked_types(raw)


    def _save_raw(self, raw, keyword=None):
        if keyword is None:
            raise ValueError('Keyword required for raw data ' + 
                             'We don\'t want to save over the original data')
        self._file_saved('Raw', keyword=keyword)
        raw.save(self._fname('raw', 'raw', 'fif', keyword),
                 verbose=False, overwrite=True)


    def _has_ICA(self, event=None, keyword=None):
        return all([op.isfile(self._fname('preprocessing', 'ica', 'fif',
                                     data_type, keyword, event))
                    for data_type in self._get_data_types()])

    def _load_ICA(self, event=None, keyword=None, data_type=None):
        from mne.preprocessing import read_ica
        fname = self._fname('preprocessing', 'ica', 'fif',
                            data_type, keyword, event)
        if op.isfile(fname):
            ica = read_ica(fname)
            self._file_loaded('ICA', event=event, data_type=data_type,
                              keyword=keyword)
            return ica
        else:
            self._no_file_error('ICA', event=event, data_type=data_type,
                                keyword=keyword)


    def _save_ICA(self, ica, event=None, keyword=None, data_type=None):
        self._file_saved('ICA', event=event, data_type=data_type,
                         keyword=keyword)
        ica.save(self._fname('preprocessing', 'ica', 'fif', data_type,
                             keyword, event))


    def _has_epochs(self, event, keyword=None):
        return op.isfile(self._fname('epochs', 'epo', 'fif', event, keyword))


    def _load_epochs(self, event, keyword=None):
        from mne import read_epochs
        if not self._has_epochs(event, keyword=keyword):
            self._no_file_error('Epochs', event=event, keyword=keyword)
        epochs = read_epochs(self._fname('epochs', 'epo', 'fif', event, keyword),
                             verbose=False, preload=True)
        self._file_loaded('Epochs', event=event, keyword=keyword)
        epochs._data = epochs._data.astype('float64') # mne bug work-around
        return self._exclude_unpicked_types(epochs)


    def _save_epochs(self, epochs, event, keyword=None):
        if keyword is None:
            raise ValueError('Keyword required for epochs data ' + 
                             'We don\'t want to save over the original data')
        self._file_saved('Epochs', event=event, keyword=keyword)
        epochs.save(self._fname('epochs', 'epo', 'fif', event, keyword))


    def _has_evoked(self, event, data_type=None, keyword=None):
        return op.isfile(self._fname('preprocessing', 'ave', 'fif',
                                     event, data_type, keyword))


    def _load_evoked(self, event, data_type=None, keyword=None):
        from mne import read_evokeds
        if not self._has_evoked(event, data_type=data_type, keyword=keyword):
            self._no_file_error('Evoked', event=event, data_type=data_type,
                                keyword=keyword)
        evoked = read_evokeds(self._fname('preprocessing', 'ave', 'fif', event,
                                          data_type, keyword),
                              verbose=False)
        self._file_loaded('Evoked', event=event, data_type=data_type,
                          keyword=keyword)
        return evoked[0]


    def _save_evoked(self, evoked, event, data_type=None, keyword=None):
        self._file_saved('Evoked', event=event, data_type=data_type,
                          keyword=keyword)
        evoked.save(self._fname('preprocessing', 'ave', 'fif', event,
                                data_type, keyword))


    def _has_autoreject(self, event, keyword=None):
        return op.isfile(self._fname('preprocessing', 'ar', 'npz',
                                     event, keyword))


    def _load_autoreject(self, event, keyword=None):
        if self._has_autoreject(event, keyword=keyword):
            f = np.load(self._fname('preprocessing', 'ar', 'npz',
                                    event, keyword))
            return f['ar'].item(), f['reject_log'].item()
        else:
            print('Autoreject must be run for ' + event)

    def _save_autoreject(self, ar, reject_log, event, keyword=None):
        np.savez_compressed(self._fname('preprocessing', 'ar', 'npz',
                                        event, keyword),
                            ar=ar, reject_log=reject_log)


    def _has_TFR(self, event, condition, value, power_type,
                 data_type=None, keyword=None):
        fname = self._fname('analyses', 'tfr', 'npy',
                            event, condition, value, power_type,
                            data_type, keyword)
        fname1b = self._fname('analyses', 'tfr_params', 'npz',
                              event, condition, value, power_type,
                              data_type, keyword)
        fname2 = self._fname('analyses', 'tfr', 'npz',
                             event, condition, value, power_type,
                             data_type, keyword)
        return ((op.isfile(fname) and op.isfile(fname1b)) or
                 op.isfile(fname2))


    def _load_TFR(self, event, condition, value, power_type,
                  data_type=None, keyword=None):
       fname = self._fname('analyses', 'tfr', 'npy',
                           event, condition, value, power_type,
                           data_type, keyword)
       fname1b = self._fname('analyses', 'tfr_params', 'npz',
                             event, condition, value, power_type,
                             data_type, keyword)
       fname2 = self._fname('analyses', 'tfr', 'npz',
                            event, condition, value, power_type,
                            data_type, keyword)
       if op.isfile(fname) and op.isfile(fname1b):
           tfr = np.load(fname)
           f = np.load(fname1b)
           frequencies, n_cycles = f['frequencies'], f['n_cycles']
       elif op.isfile(fname2):
           f = np.load(fname2)
           tfr, frequencies, n_cycles = f['tfr'], f['frequencies'], f['n_cycles']
       else:
           self._no_file_error('TFR', power_type, event=event,
                               condition=condition, value=value,
                               data_type=data_type, keyword=keyword)
       self._file_loaded('TFR', power_type, event=event,
                         condition=condition, value=value,
                         data_type=data_type, keyword=keyword)
       return tfr, frequencies, n_cycles


    def _save_TFR(self, tfr, frequencies, n_cycles,
                  event, condition, value, power_type,
                  data_type=None, keyword=None, compressed=True):
        self._file_saved('TFR', power_type, event=event,
                         condition=condition, value=value,
                         data_type=data_type, keyword=keyword)
        fname1 = self._fname('analyses', 'tfr', 'npy', event, condition,
                             value, power_type, data_type, keyword)
        fname1b = self._fname('analyses', 'tfr_params', 'npz', event,
                              condition, value, power_type, data_type,
                              keyword)
        fname2 = self._fname('analyses', 'tfr', 'npz', event,
                             condition, value, power_type, data_type,
                             keyword)
        if compressed:
            np.savez_compressed(fname2, tfr=tfr, frequencies=frequencies,
                                n_cycles=n_cycles)
            if op.isfile(fname1) and op.isfile(fname1b):
                os.remove(fname1)
                os.remove(fname1b)
        else:
            np.save(fname1, tfr)
            np.savez_compressed(fname1b, frequencies=frequencies,
                                n_cycles=n_cycles)
            if op.isfile(fname2):
                os.remove(fname2)


    def _has_CPT(self, event, condition, value, band=None,
                 data_type=None, keyword=None):
        return op.isfile(self._fname('analyses', 'CPT', 'npz',
                                     event, condition, value,
                                     band, data_type, keyword))


    def _load_CPT(self, event, condition, value, band=None,
                  data_type=None, keyword=None):
        if self._has_CPT(event, condition, value, band=band,
                         data_type=data_type, keyword=keyword):
            f = np.load(self._fname('analyses', 'CPT', 'npz',
                                    event, condition, value, band,
                                    data_type, keyword))
            self._file_loaded('Cluster Permutation Test', event=event,
                              condition=condition, value=value,
                              data_type=data_type, keyword=keyword)
            return (f['clusters'], f['cluster_p_values'], f['times'],
                    f['ch_dict'].item(), f['frequencies'], f['band'])
        else:
            self._no_file_error('Cluster Permutation Test', band,
                                event=event, condition=condition, value=value,
                                data_type=data_type, keyword=keyword)


    def _save_CPT(self, clusters, cluster_p_values, times, ch_dict, frequencies,
                  band, event, condition, value, data_type=None, keyword=None):
        self._file_saved('Cluster Permutation Test', band,
                         event=event, condition=condition, value=value,
                         data_type=data_type, keyword=keyword)
        fname = self._fname('analyses', 'CPT', 'npz', event, condition, value,
                            band, data_type, keyword)
        np.savez_compressed(fname, clusters=clusters,
                            ch_dict=ch_dict, times=times,
                            frequencies=(frequencies if 
                                frequencies is not None else []),
                            cluster_p_values=cluster_p_values,
                            band=band if band is not None else [])


    def _has_inverse(self, event, condition, value, keyword=None):
        fname = self._fname('source_estimates', 'inv', 'fif', keyword,
                            event, condition, value)
        fname2 = self._fname('source_estimates', 'inverse_params', 'npz',
                             keyword, event, condition, value)
        return op.isfile(fname) and op.isfile(fname2)


    def _load_inverse(self, event, condition, value, keyword=None):
        from mne.minimum_norm import read_inverse_operator
        if self._has_inverse(event, condition, value, keyword=keyword):
            fname = self._fname('source_estimates', 'inv', 'fif', keyword,
                                event, condition, value)
            fname2 = self._fname('source_estimates', 'inverse_params', 'npz',
                                 keyword, event, condition, value)
            f = np.load(fname2)
            return (read_inverse_operator(fname), f['lambda2'].item(),
                    f['method'].item(), f['pick_ori'].item())
        else:
            self._no_file_error('Inverse', event=event, condition=condition,
                                value=value, keyword=keyword)


    def _save_inverse(self, inv, lambda2, method, pick_ori,
                      event, condition, value, keyword=None):
        from mne.minimum_norm import write_inverse_operator
        self._file_saved('Inverse', event=event, condition=condition,
                         value=value, keyword=keyword)
        write_inverse_operator(self._fname('source_estimates', 'inv', 'fif',
                                           keyword, event, condition, value),
                               inv, verbose=False)
        np.savez_compressed(self._fname('source_estimates', 'inverse_params', 'npz',
                                        keyword, event, condition, value),
                            lambda2=lambda2, method=method, pick_ori=pick_ori)


    def _has_source(self, event, condition, value, keyword=None):
        fname = self._fname('source_estimates', 'source-lh', 'stc', keyword,
                            event, condition, value)
        return op.isfile(fname)


    def _load_source(self, event, condition, value, keyword=None):
        from mne import read_source_estimate
        fname = self._fname('source_estimates', 'source-lh', 'stc', keyword,
                            event, condition, value)
        if self._has_source(event, condition, value, keyword=keyword):
            self._file_loaded('Source Estimate', event=event, condition=condition,
                              value=value, keyword=keyword)
            return read_source_estimate(fname)
        else:
            print('Source not found for %s %s %s' % (event, condition, value))


    def _save_source(self, stc, event, condition, value, keyword=None):
        self._file_saved('Source Estimate', event=event, condition=condition,
                          value=value, keyword=keyword)
        stc.save(self._fname('source_estimates', 'source', None, keyword,
                             event, condition, value),
                 ftype='stc')


    def _has_PSD(self, keyword):
        return op.isfile(self._fname('analyses', 'psd', 'npz', keyword))

    def _load_PSD(self, keyword):
        if self._has_PSD(keyword):
            fname = self._fname('analyses', 'psd', 'npz', keyword)
            self._file_loaded('Power Spectral Density', keyword=keyword)
            return np.load(fname)['image']
        else:
            return None

    def _save_PSD(self, image, keyword):
        self._file_saved('Power Spectral Density', keyword=keyword)
        np.savez_compressed(self._fname('analyses', 'psd', 'npz', keyword),
                            image=image)

    def _file_message(self, action, ftype,*tags, event=None,
                      condition=None, value=None,
                      data_type=None, keyword=None):
        return ('%s for %s' % (action, ftype) +
                (' %s' % event)*(event is not None) + 
                (' %s' % condition)*(condition is not None) + 
                (' %s' % value)*(value is not None) +
                (' %s' % data_type)*(data_type is not None) +
                (' %s' % keyword)*(keyword is not None) + 
                (' %s' % ' '.join([str(tag) for tag in tags if 
                                  tag and tag is not None])))


    def _file_loaded(self, ftype,*tags, event=None, condition=None,
                     value=None, data_type=None, keyword=None):
        print(self._file_message('File loaded', ftype, event=event,
                                 condition=condition,
                                 value=value, data_type=data_type,
                                 keyword=keyword,*tags))


    def _file_saved(self, ftype,*tags, event=None, condition=None, value=None,
                    data_type=None, keyword=None):
        print(self._file_message('File saved', ftype, event=event,
                                 condition=condition,
                                 value=value, data_type=data_type,
                                 keyword=keyword,*tags))


    def _no_file_error(self, ftype,*tags, event=None, condition=None,
                       value=None, data_type=None, keyword=None):
        raise ValueError(self._file_message('No file found',
                                 ftype, event=event,
                                 condition=condition,
                                 value=value, data_type=data_type,
                                 keyword=keyword,*tags))


    def remove(self, event=None, keyword=None):
        dir_name = 'raw' if event is None else 'epochs'
        suffix = 'epo' if event else 'raw'
        fname = self._fname(dir_name, suffix, 'fif', event, keyword)
        if op.isfile(fname):
            print('Deleting %s' % fname)
            os.remove(fname)


    def _overwrite_error(self, ftype,*tags, event=None, condition=None,
                         values=None, keyword=None):
        error_msg = ('%s already exists for %s' % (ftype, event) +
                     (' %s' % ' '.join([str(tag) for tag in tags])) + 
                     (' %s' % keyword)*(keyword is not None) +
                     (' %s' % condition)*(condition is not None) +
                     (' %s' % (' '.join([str(v) for v in values]))
                      if values is not None else '') +
                     ', use \'overwrite=True\' to overwrite')
        raise ValueError(error_msg)


    def _get_data_types(self):
        return (['grad', 'mag']*self.meg + ['eeg']*self.eeg + 
                ['ecog']*self.ecog + ['seeg']*self.seeg)


    def preprocess(self, event=None):
        # preprocessing
        self.autoMarkBads()
        self.findICA()
        for event in self.events:
            self.makeEpochs(event)
            self.markAutoReject(event)

    def getEvents(self, baseline=True):
        return (list(self.events.keys()) + ['Response']*(self.response is not None) +
                ['Baseline']*(self.baseline is not None and baseline))

    def _default_aux(self, inst, eogs=None, ecgs=None):
        from mne import pick_types
        if eogs is None:
            inds = pick_types(inst.info, meg=False, eog=True)
            eogs = [inst.ch_names[ind] for ind in inds]
            print('Using ' + (' '.join(eogs) if eogs else 'no channels') + ' as eogs')
        if ecgs is None:
            inds = pick_types(inst.info, meg=False, ecg=True)
            ecgs = [inst.ch_names[ind] for ind in inds]
            print('Using ' + (' '.join(ecgs) if ecgs else 'no channels') + ' as ecgs')
        return eogs, ecgs

    def _combine_insts(self, insts):
        from mne import EpochsArray
        from mne.io import RawArray, BaseRaw
        if len(insts) < 1:
            raise ValueError('Nothing to combine')
        inst_data = insts[0]._data
        inst_info = insts[0].info
        for inst in insts[1:]:
            inst_data = np.concatenate([inst_data, inst._data], axis=-2)
            inst_info['ch_names'] += inst.info['ch_names']
            inst_info['chs'] += inst.info['chs']
            inst_info['nchan'] += inst.info['nchan']
            inst_info['bads'] += inst.info['bads']

        if isinstance(insts[0], BaseRaw):
            return RawArray(inst_data, inst_info).set_annotations(insts[0].annotations)
        else:
            return EpochsArray(inst_data, inst_info, events=insts[0].events, tmin=insts[0].tmin)

    def findICA(self, event=None, keyword_in=None, keyword_out=None,
                eogs=None, ecgs=None, n_components=None, l_freq=None, h_freq=40,
                detrend=1, component_optimization_n=3, tmin=None, tmax=None,
                vis_tmin=None, vis_tmax=None, seed=11, overwrite=False,
                overwrite_ica=False):
        # keyword_out functionality was added so that ICA can be computed on
        # one raw data and applied to another
        # note: filter only filters evoked
        from mne.io import BaseRaw
        from mne.preprocessing import ICA

        keyword_out = keyword_in if keyword_out is None else keyword_out
        data_types = self._get_data_types()

        if self._has_ICA(event=event, keyword=keyword_out) and not overwrite_ica:
            raise ValueError('Use \'overwrite_ica=True\' to overwrite')
        if event is None:
            if self._has_raw(keyword=keyword_out) and not overwrite:
                self._overwrite_error('Raw', keyword=keyword_out)
            inst = self._load_raw(keyword=keyword_in)
        else:
            if self._has_epochs(event, keyword=keyword_out) and not overwrite:
                self._overwrite_error('Epochs', event=event, keyword=keyword_out)
            inst = self._load_epochs(event, keyword=keyword_in)
            tmin, tmax = self._default_t(event, tmin, tmax)
            inst = inst.crop(tmin=tmin, tmax=tmax)
        eogs, ecgs = self._default_aux(inst, eogs, ecgs)
        if not all([ch in inst.ch_names for ch in eogs + ecgs]):
            raise ValueError('Auxillary channels not in channel list.')
        ica_insts = []
        for dt in data_types:
            print(dt)
            inst2 = self._pick_types(inst, dt)
            ica = ICA(method='fastica', n_components=n_components,
                      random_state=seed)
            ica.fit(inst2)
            fig = ica.plot_components(picks=np.arange(ica.get_components().shape[1]),
                                      show=False)
            fig.savefig(self._fname('plots', 'components', 'jpg', dt, keyword_out))
            plt.close(fig)
            #
            if isinstance(inst, BaseRaw):
                raw = self._pick_types(inst, dt, aux=True)
                all_scores = self._make_ICA_components(raw, ica, eogs, ecgs, detrend,
                                                       l_freq, h_freq, dt, keyword_out,
                                                       vis_tmin, vis_tmax)
            '''if component_optimization_n:
                ica = self._optimize_components(raw, ica, all_scores,
                                                component_optimization_n,
                                                keyword_in, kw)'''
            inst2 = ica.apply(inst2, exclude=ica.exclude)
            self._save_ICA(ica, event=event, data_type=dt, keyword=keyword_out)
            ica_insts.append(inst2)
        try:
            ica_insts.append(inst.copy().pick_types(meg=False, eeg=False, eog=True,
                                                    ecg=True, stim=True, exclude=[]))
        except Exception as e:
            print('No eog, ecg or stim channels', e)
        inst = self._combine_insts(ica_insts)

        if isinstance(inst, BaseRaw):
            self._save_raw(inst, keyword=keyword_out)
        else:
            self._save_epochs(inst, event, keyword=keyword_out)

    def _optimize_components(self, raw, ica, all_scores,
                             component_optimization_n, keyword, data_type):
        # get component_optimization_n of components
        components = []
        for ch in all_scores:
            current_scores = list(all_scores[ch])
            for n in range(component_optimization_n):
                component = current_scores.index(max(current_scores))
                components.append(component)
                current_scores.pop(component)
        # get ways to combine the components
        def int2bin(n, i):
            b = []
            for j in range(n-1,-1,-1):
                if i/(2**j):
                    i -= 2**j
                    b.append(1)
                else:
                    b.append(0)
            return list(reversed(b))
        combinations = [int2bin(len(components), i) for i in range(2**len(components))]
        min_score = None
        evokeds = {}
        for ch in all_scores:
            evokeds[ch] = self._load_evoked('ica_%s' % ch, data_type=data_type,
                                            keyword=keyword)
        print('Testing ICA component combinations for minimum correlation to artifact epochs')
        for combo in tqdm(combinations):
            score = 0
            ica.exclude = [component for i, component in enumerate(components) if combo[i]]
            for ch in all_scores:
                evoked = ica.apply(evokeds[ch].copy(), exclude=ica.exclude)
                sfreq = int(evoked.info['sfreq'])
                evoked_data = evoked.data[:, sfreq/10:-sfreq/10]
                for i in range(evoked_data.shape[0]):
                    evoked_data[i] -= np.median(evoked_data[i])
                score += abs(evoked_data).sum()*evoked_data.std(axis=0).sum()
            if min_score is None or score < min_score:
                best_combo = combo
                min_score = score
        ica.exclude = [component for i, component in enumerate(components) if best_combo[i]]
        return ica

    def _make_ICA_components(self, raw, ica, eogs, ecgs, detrend, l_freq, h_freq,
                             data_type, keyword, vis_tmin, vis_tmax):
        from mne.preprocessing import create_eog_epochs, create_ecg_epochs
        if vis_tmin is not None:
            raw = raw.copy().crop(tmin=vis_tmin)
        if vis_tmax is not None:
            raw = raw.copy().crop(tmax=vis_tmax)
        all_scores = {}
        for ch in eogs:
            try:
                epochs = create_eog_epochs(raw, ch_name=ch, h_freq=8)
                indices, scores = ica.find_bads_eog(epochs, ch_name=ch)
                all_scores[ch] = scores
            except:
                print('EOG %s dead' % ch)
                continue
            if l_freq is not None or h_freq is not None:
                epochs = epochs.filter(l_freq=l_freq, h_freq=h_freq)
            evoked = epochs.average()
            if detrend is not None:
                evoked = evoked.detrend(detrend)
            self._save_evoked(evoked, 'ica_%s' % ch,
                              data_type=data_type, keyword=keyword)
            self._exclude_ICA_components(ica, ch, indices, scores)

        for ecg in ecgs:
            try:
                epochs = create_ecg_epochs(raw, ch_name=ecg)
                indices, scores = ica.find_bads_ecg(epochs)
                all_scores[ecg] = scores
            except:
                print('ECG %s dead' % ecg)
                continue
            if l_freq is not None or h_freq is not None:
                epochs = epochs.filter(l_freq=l_freq, h_freq=h_freq)
            evoked = epochs.average()
            if detrend is not None:
                evoked = evoked.detrend(detrend)
            self._save_evoked(evoked, 'ica_%s' % ecg,
                              data_type=data_type, keyword=keyword)
            self._exclude_ICA_components(ica, ecg, indices, scores)
            return all_scores


    def _exclude_ICA_components(self, ica, ch, indices, scores):
        for ind in indices:
            if ind not in ica.exclude:
                ica.exclude.append(ind)
        print('Components removed for %s: ' % ch +
              ' '.join([str(i) for i in indices]))
        fig = ica.plot_scores(scores, exclude=indices, show=False)
        fig.savefig(self._fname('plots', 'source_scores', 'jpg', ch))
        plt.close(fig)


    def plotICA(self, event=None, keyword_in=None, keyword_out=None,
                eogs=None, ecgs=None, tmin=None, tmax=None,
                ylim=dict(eeg=[-40, 40], grad=[-400, 400], mag=[-1000, 1000]),
                plot_properties=False, show=True):
        from mne.io import BaseRaw
        keyword_out = keyword_in if keyword_out is None else keyword_out
        if event is None:
            inst = self._load_raw(keyword=keyword_in)
        else:
            inst = self._load_epochs(event, keyword=keyword_in)
            tmin, tmax = self._default_t(event, tmin, tmax)
            inst = inst.crop(tmin=tmin, tmax=tmax)
        eogs, ecgs = self._default_aux(inst, eogs, ecgs)
        data_types = self._get_data_types()
        ica_insts = []
        for dt in data_types:
            inst1b = self._pick_types(inst, dt)
            inst2 = self._pick_types(inst, dt)
            ica = self._load_ICA(event=event, data_type=dt,
                                 keyword=keyword_out)
            if isinstance(inst, BaseRaw):
                for ch in eogs:
                    try:
                        evoked = self._load_evoked('ica_%s' % ch,
                                                   data_type=dt,
                                                   keyword=keyword_out)
                        self._plot_ICA_sources(ica, evoked, ch, show)
                    except:
                        print('%s dead/not working' % ch)
                for ecg in ecgs:
                    try:
                        evoked = self._load_evoked('ica_%s' % ecg,
                                                   data_type=dt,
                                                   keyword=keyword_out)
                        self._plot_ICA_sources(ica, evoked, ecg, show)
                    except:
                        print('%s dead/not working' % ecg)
            fig = ica.plot_components(picks=np.arange(ica.get_components().shape[1]),
                                      show=False)
            fig.show()
            ica.plot_sources(inst2, block=show, show=show, title=self.subject)
            inst2 = ica.apply(inst2, exclude=ica.exclude)
            if isinstance(inst, BaseRaw):
                for ch in eogs:
                    try:
                        evoked = self._load_evoked('ica_%s' % ch,
                                                   data_type=dt,
                                                   keyword=keyword_out)
                        self._plot_ICA_overlay(ica, evoked, ch, show)
                    except:
                        print('%s dead/not working' % ch)
                for ecg in ecgs:
                    try:
                        evoked = self._load_evoked('ica_%s' % ecg,
                                                   data_type=dt,
                                                   keyword=keyword_out)
                        self._plot_ICA_overlay(ica, evoked, ecg, show)
                    except:
                        print('%s dead/not working' % ecg)
            else:
                fig = inst1b.average().plot(show=False, ylim=ylim,
                                            window_title='Before ICA')
                self._show_fig(fig, show)
                fig2 = inst2.average().plot(show=False, ylim=ylim,
                                            window_title='After ICA')
                self._show_fig(fig2, show)
            if plot_properties:
                self._plot_ICA_properties(inst1b, ica, ica.exclude, show)
            plt.show()
            ica_insts.append(inst2)
            self._save_ICA(ica, data_type=dt, keyword=keyword_out)
        try:
            ica_insts.append(inst.copy().pick_types(meg=False, eeg=False, eog=True,
                                                    ecg=True, stim=True, exclude=[]))
        except Exception as e:
            print('No eog, ecg or stim channels', e)
        inst = self._combine_insts(ica_insts)

        if isinstance(inst, BaseRaw):
            self._save_raw(inst, keyword=keyword_out)
        else:
            self._save_epochs(inst, event, keyword=keyword_out)


    def _plot_ICA_overlay(self, ica, evoked, ch, show):
        evoked = evoked.detrend(1)
        fig = ica.plot_overlay(evoked, show=False)
        fig.suptitle('%s %s' % (self.subject, ch))
        fig.savefig(self._fname('plots', 'ica_overlay', 'jpg', ch))
        self._show_fig(fig, show)


    def _plot_ICA_sources(self, ica, evoked, ch, show):
        fig = ica.plot_sources(evoked, exclude=ica.exclude, show=False)
        fig.suptitle('%s %s' % (self.subject, ch))
        fig.savefig(self._fname('plots', 'ica_time_course', 'jpg', ch))
        self._show_fig(fig, show)


    def _plot_ICA_properties(self, inst, ica, picks, show):
        figs = ica.plot_properties(inst, picks=picks, show=False)
        for i, fig in enumerate(figs):
            fig.suptitle('%s' % self.subject)
            fig.savefig(self._fname('plots',
                                    'ica_propterties',
                                    'jpg', picks[i]))
            self._show_fig(fig, show)


    def autoMarkBads(self, keyword_in=None, keyword_out=None,
                     flat=dict(grad=1e-11, # T / m (gradiometers)
                               mag=5e-13, # T (magnetometers)
                               eeg=2e-5, # V (EEG channels)
                               ),
                     reject=dict(grad=5e-10, # T / m (gradiometers)
                                 mag=1e-11, # T (magnetometers)
                                 eeg=5e-4, # V (EEG channels)
                                 ),
                     bad_seeds=0.25, seeds=1000, datalen=1000,
                     overwrite=False):
        # now we will use seeding to remove bad channels
        keyword_out = keyword_out if not keyword_out is None else keyword_in
        if (os.path.isfile(self._fname('raw', 'raw', 'fif', keyword_out)) and
            not overwrite):
           print('Raw data already marked for bads, use \'overwrite=True\'' +
                 ' to recalculate.')
           return
        raw = self._load_raw(keyword=keyword_in)
        raw.info['bads'] = []
        data_types = self._get_data_types()
        rawlen = len(raw._data[0])
        for dt in data_types:
            print(dt)
            raw2 = self._pick_types(raw, dt)
            for i in range(len(raw2.ch_names)):
                flat_count, reject_count = 0, 0
                for j in range(seeds):
                    start = np.random.randint(0, rawlen-datalen)
                    seed = raw2._data[i, start:start+datalen]
                    min_c = seed.min()
                    max_c = seed.max()
                    diff_c = max_c - min_c
                    if diff_c < flat[dt]:
                        flat_count += 1
                    if diff_c > reject[dt]:
                        reject_count += 1
                if flat_count > (seeds * bad_seeds):
                    raw.info['bads'].append(raw2.ch_names[i])
                    print(raw2.ch_names[i] + ' removed: flat')
                elif reject_count > (seeds * bad_seeds):
                    raw.info['bads'].append(raw2.ch_names[i])
                    print(raw2.ch_names[i] + ' removed: reject')

        self._save_raw(raw, keyword=keyword_out)


    def closePlots(self):
        plt.close('all')


    def plotRaw(self, n_per_screen=20, scalings=None, keyword_in=None,
                keyword_out=None, l_freq=0.5, h_freq=40, overwrite=False):
        from mne import pick_types
        keyword_out = keyword_in if keyword_out is None else keyword_out
        if (os.path.isfile(self._fname('raw', 'raw', 'fif', keyword_out))
            and not overwrite):
            print('Use \'overwrite = True\' to overwrite')
            return
        raw = self._load_raw(keyword=keyword_in)
        bads_ind = [raw.info['ch_names'].index(ch) for ch in raw.info['bads']]
        this_chs_ind = list(pick_types(raw.info, meg=self.meg, eeg=self.eeg)) + bads_ind
        aux_chs_ind = list(pick_types(raw.info, meg=False, eog=True, ecg=True))
        order = []
        n = n_per_screen-len(aux_chs_ind)
        for i in range(len(this_chs_ind)//n+1):
            order.append(this_chs_ind[i*n:min([len(this_chs_ind),(i+1)*n])])
            order.append(aux_chs_ind)
        order = np.array(np.concatenate(order), dtype=int)
        if self.eeg:
            raw.set_eeg_reference(ref_channels=[], projection=False)
        elif self.meg:
            order = None
        if l_freq is not None or h_freq is not None:
            raw2 = raw.copy().filter(l_freq=l_freq, h_freq=h_freq)
        else:
            raw2 = raw.copy()
        raw2.plot(show=True, block=True, color=dict(eog='steelblue'),
                  title="%s Bad Channel Selection" % self.subject, order=order,
                  scalings=scalings)
        raw.info['bads'] = raw2.info['bads']
        self._save_raw(raw, keyword=keyword_out)


    def interpolateBads(self, event=None, keyword_in=None, keyword_out=None):
        keyword_out = keyword_in if keyword_out is None else keyword_out
        if event is None:
            raw = self._load_raw(keyword=keyword_in)
            raw = raw.interpolate_bads(reset_bads=True)
            self._save_raw(raw, keyword=keyword_out)
        else:
            epo = self._load_epochs(event, keyword=keyword_in)
            epo = epo.interpolate_bads(reset_bads=True)
            self._save_epochs(event, keyword=keyword_out)


    def downsample(self, event=None, keyword_in=None, keyword_out=None,
                   new_sfreq=200, npad='auto', window='boxcar', n_jobs=1,
                   overwrite=False):
        keyword_out = keyword_in if keyword_out is None else keyword_out
        if event is None:
            if self._has_raw(keyword=keyword_out) and not overwrite:
                self._overwrite_error('Raw', keyword=keyword_out)
            raw = self._load_raw(keyword=keyword_in)
            raw = raw.resample(new_sfreq, npad=npad, window=window, n_jobs=n_jobs)
            self._save_raw(raw, keyword=keyword_out)
        else:
            if self._has_epochs(event, keyword=keyword_out) and not overwrite:
                self._overwrite_error('Epochs', event=event, keyword=keyword_out)
            epochs = self._load_epochs(event, keyword=keyword_in)
            epochs = epochs.resample(new_sfreq, npad=npad, window=window,
                                     n_jobs=n_jobs)
            self._save_epochs(epochs, event, keyword=keyword_out)


    def makeEpochs(self, keyword_in=None, keyword_out=None, detrend=0,
                   normalized=True, overwrite=False):
        from mne import EpochsArray
        keyword_out = keyword_in if keyword_out is None else keyword_out
        if (all([self._has_epochs(event, keyword_out) for event in self.getEvents()])
            and not overwrite):
            self._overwrite_error('Epochs', keyword=keyword_out)
        raw = self._load_raw(keyword=keyword_in)

        n_events = None

        if self.baseline:
            ch, tmin, tmax = self.baseline
            n_events = self._make_epochs(raw, 'Baseline', ch, tmin, tmax, detrend,
                                         keyword_out, n_events=n_events)

        if normalized:
            bl_epochs = self._load_epochs('Baseline', keyword=keyword_out)
            baseline_data = bl_epochs.crop(tmin=tmin, tmax=tmax).get_data()
            baseline_arr = baseline_data.mean(axis=2)

        for event in self.events:
            ch, tmin, tmax = self.events[event]
            n_events = self._make_epochs(raw, event, ch, tmin, tmax, detrend,
                                         keyword_out, n_events=n_events)
            if normalized:
                epochs = self._load_epochs(event, keyword=keyword_out)
                epochs_data = epochs.get_data()
                epochs_demeaned_data = np.array([arr - baseline_arr.T
                                                 for arr in epochs_data.T]).T
                epochs = EpochsArray(epochs_demeaned_data, epochs.info,
                                     events=epochs.events, verbose=False,
                                     proj=False, tmin=tmin-self.tbuffer)
        if self.response:
            ch, tmin, tmax = self.response
            self._make_epochs(raw, 'Response', ch, tmin, tmax, detrend, keyword_out,
                              n_events=n_events, response=True)

    def _find_events(self, raw, ch): #backward compatible
        from mne import find_events, events_from_annotations
        if isinstance(ch, tuple):
            ch, event_id = ch
        else:
            event_id = None
        try:
            events = find_events(raw, stim_channel=ch, output="onset", verbose=False)
            if event_id is not None:
                events = events[np.where(events[:, 2]==event_id)[0]]
        except:
            events, event_id2 = events_from_annotations(raw)
            if event_id is None:
                events = events[np.where(events[:, 2]==event_id2[ch])[0]]
            else:
                events = events[np.where(events[:, 2]==event_id)[0]]
        return events

    def _make_epochs(self, raw, event, ch, tmin, tmax, detrend, keyword_out, n_events=None,
                     response=False):
        from mne import Epochs
        try:
            events = self._find_events(raw, ch)
        except:
            raise ValueError('%s channel not found in raw' % event +
                             ', maybe you meant to use normalized=False' * (event == 'Baseline'))
        n_events2 = len(events)
        print('%s events found: %i' % (event, n_events2))
        if response:
            response_events = np.setdiff1d(np.arange(n_events), self.no_response)
            exclude = np.intersect1d(response_events, self.exclude_response)
            if n_events is not None and n_events2 + len(self.no_response) != n_events:
                raise ValueError('%i events compared to ' % n_events +
                                 '%i responses + %i excluded responses ' % (n_events2, diff) +
                                 'doesn\'t add up')
            events[:, 2] = response_events
        else:
            exclude = self.exclude_response
            if n_events is not None and n_events2 != n_events:
                raise ValueError('%i events from previous stimuli, ' % n_events +
                                 '%i events from %s' % (n_events2, event))
            events[:, 2] = np.arange(n_events2)
        events = np.delete(events, exclude, axis=0)

        epochs = Epochs(raw, events, tmin=tmin-self.tbuffer,
                        tmax=tmax+self.tbuffer, baseline=None, verbose=False,
                        detrend=detrend, preload=True)
        if self.eeg:
            epochs = epochs.set_eeg_reference(ref_channels='average',
                                              projection=False)
        self._save_epochs(epochs, event, keyword=keyword_out)
        return n_events2


    def plotEpochs(self, event, n_epochs=20, n_channels=20, scalings=None,
                   tmin=None, tmax=None, l_freq=None, h_freq=None,
                   keyword_in=None, keyword_out=None, overwrite=False):
        # note: if linear trend, apply l_freq filter
        keyword_out = keyword_in if keyword_out is None else keyword_out
        if self._has_epochs(event, keyword=keyword_out) and not overwrite:
            self._overwrite_error('Epochs', event=event, keyword=keyword_out)
        epochs = self._load_epochs(event, keyword=keyword_in)
        tmin, tmax = self._default_t(event, tmin, tmax)
        epochs_copy = epochs.copy().crop(tmin=tmin, tmax=tmax)
        if l_freq is not None or h_freq is not None:
            epochs_copy = epochs_copy.filter(l_freq=l_freq, h_freq=h_freq)
        epochs_copy.plot(n_epochs=n_epochs, n_channels=n_channels, block=True,
                             scalings=scalings)
        epochs.info['bads'] = epochs_copy.info['bads']
        drop_indices = []
        i = 0
        for e1, e2 in zip(epochs.drop_log, epochs_copy.drop_log):
            if e1 == []:
                if e2 != []:
                    drop_indices.append(i)
                i += 1

        epochs = epochs.drop(drop_indices)
        self._save_epochs(epochs, event, keyword=keyword_out)


    def plotTopo(self, event, condition=None, values=None,
                 epochs=None, keyword=None, ylim={'eeg':[-30, 30]},
                 l_freq=None, h_freq=None, tmin=None, tmax=None, detrend=1,
                 seed=11, downsample=True, show=True):
        epochs = self._prepare_epochs(event, epochs, keyword, tmin, tmax, l_freq, h_freq)
        if condition is None:
            values = ['all']
            value_indices = {'all':[]}
        else:
            values = self._default_values(condition, values=values)
            value_indices = self._get_indices(epochs, condition, values)
            if downsample:
                np.random.seed(seed)
                nTR = min([len(value_indices[value]) for value in value_indices])
        fig, axs = plt.subplots((2*self.meg+self.eeg), len(values),
                               figsize=(5*len(values), 5*(2*self.meg+self.eeg)))
        if self.eeg and not self.meg:
            if not type(axs) is np.ndarray:
                axs = np.array([axs])
            axs = axs[np.newaxis,:]
        fig.suptitle('%s %s %s' % (self.subject, event, condition))
        if not isinstance(axs, np.ndarray):
            axs = np.array([axs])
        for i, value in enumerate(values):
            if not value in value_indices:
                continue
            indices = value_indices[value]
            if condition is not None and downsample:
                print('Subsampling %i/%i %s %s.' % (nTR, len(indices), condition,
                                                    value))
                np.random.shuffle(indices)
                indices = indices[:nTR]
            if value == 'all':
                evoked = epochs.average()
            else:
                evoked = epochs[indices].average()
            if detrend:
                evoked = evoked.detrend(order=detrend)
            if self.meg:
                ax = axs[0, i]
                if i == 0:
                    ax.set_ylabel('mag')
                evoked.copy().pick_types(meg='mag').plot_topo(axes=ax,
                                                        show=False, ylim=ylim)
                ax2 = axs[1, i]
                if i == 0:
                    ax2.set_ylabel('grad')
                evoked.copy().pick_types(meg='grad').plot_topo(axes=ax2,
                                                        show=False, ylim=ylim)
            if self.eeg:
                ax = axs[2*self.meg, i]
                if i == 0:
                    ax.set_ylabel('eeg')
                evoked.copy().pick_types(eeg=True).plot_topo(axes=ax,
                                                             show=False,
                                                             ylim=ylim)
            ax = axs[0, i]
            ax.set_title(value)
        fname = self._fname('plots', 'evoked', 'jpg', keyword, event, condition,
                            *value_indices.keys())
        fig.savefig(fname)
        self._show_fig(fig, show)


    def plotTopomapBands(self, event, condition, values=None, keyword=None,
                         tfr_keyword=None, power_type='npl', contrast=False,
                         tmin=None, tmax=None, tfr=True,
                         bands={'theta':(4, 8), 'alpha':(8, 15),
                                'beta':(15, 35), 'low-gamma':(35, 80),
                                'high-gamma':(80, 150)},
                         vmin=None, vmax=None, contours=6, time_points=5, show=True):
        for band in bands:
            self.plotTopomap(event, condition, values=values, keyword=keyword,
                             contrast=contrast, tmin=tmin, tmax=tmax, tfr=True,
                             tfr_keyword=tfr_keyword, power_type=power_type,
                             band_struct=(band, bands[band][0], bands[band][1]),
                             vmin=vmin, vmax=vmax, contours=contours,
                             time_points=time_points, show=show)


    def plotTopomap(self, event, condition, values=None, keyword=None,
                    tfr_keyword=None, power_type='npl', contrast=False,
                    tmin=None, tmax=None, tfr=False, band_struct=None,
                    vmin=None, vmax=None, contours=6, time_points=5,
                    show=True):
        from mne.time_frequency import AverageTFR
        from mne import EvokedArray
        epochs = self._load_epochs(event, keyword=keyword)
        values = self._default_values(condition, values, contrast)
        value_indices = self._get_indices(epochs, condition, values)
        tmin, tmax = self._default_t(event, tmin, tmax)
        epochs = epochs.crop(tmin=tmin, tmax=tmax)
        times = epochs.times
        info = epochs.info
        band_title = '%s ' % (band_struct[0] if band_struct is not None else '')
        if tfr:
            values_dict, frequencies = \
                self._get_tfr_data(event, condition, values, tfr_keyword,
                                   power_type, value_indices,
                                   band=band_struct, mean_and_std=False,
                                   band_mean=False)
        else:
            values_dict = self._get_data(epochs, values, value_indices,
                                         mean_and_std=False)
            frequencies = None
        if contrast:
            fig, axes = plt.subplots(1, time_points+(not tfr))
            fig.suptitle(band_title + '%s %s Contrast' % (values[0], values[1]))
            epochs_0 = values_dict[values[0]]
            epochs_1 = values_dict[values[1]]
            if tfr:
                nave = min([epochs_0.shape[0], epochs_1.shape[0]])
                epochs_0 = np.swapaxes(epochs_0, 2, 3)
                epochs_1 = np.swapaxes(epochs_1, 2, 3)
                tfr_con_data = epochs_1.mean(axis=0) - epochs_0.mean(axis=0)
                evo_con = AverageTFR(info, tfr_con_data, times, frequencies, nave)
                dt = (tmax-tmin)/time_points
                for i, t in enumerate(np.linspace(tmin, tmax, time_points)):
                    evo_con.plot_topomap(colorbar=True if i == time_points-1 else False,
                                     vmin=vmin, vmax=vmax, contours=contours, axes=axes[i],
                                     title='time=%0.1f' % t, tmin=t-dt/2, tmax=t+dt/2, show=False)
            else:
                evo_con_data = evo_1.mean(axis=0) - evo_0.mean(axis=0)
                evo_con = EvokedArray(evo_con_data, info, tmin=tmin)
                evo_con.plot_topomap(colorbar=True, vmin=vmin, vmax=vmax,
                                     contours=contours, axes=axes, show=False)
            fig.savefig(self._fname('plots', 'topo', 'jpg', event, condition,
                                    values[0], values[1],
                                    '' if band_struct is None else band_struct[0]))
            self._show_fig(fig, show)
        else:
            for i, value in enumerate(values):
                fig, axes = plt.subplots(1, time_points+(not tfr))
                fig.suptitle(band_title + '%s %s' % (condition, value))
                epochs_data = values_dict[value]
                if tfr:
                    nave = epochs_data.shape[0]
                    evo_data = np.swapaxes(epochs_data, 2, 3).mean(axis=0)
                    evo = AverageTFR(info, evo_data, times, frequencies, nave)
                    dt = (tmax-tmin)/time_points
                    for i, t in enumerate(np.linspace(tmin, tmax, time_points)):
                        evo.plot_topomap(colorbar=True if i == time_points-1 else False,
                                         vmin=vmin, vmax=vmax, contours=contours, axes=axes[i],
                                         title='time=%0.1f' % t, tmin=t-dt/2, tmax=t+dt/2, show=False)
                else:
                    evo = EvokedArray(epochs_data.mean(axis=0), info, tmin=tmin)
                    evo.plot_topomap(colorbar=True, vmin=vmin, vmax=vmax,
                                     contours=contours, axes=axes, show=False)
                fig.savefig(self._fname('plots', 'topo', 'jpg', event, condition,
                                        value, '' if band_struct is None else band_struct[0]))
                self._show_fig(fig, show)


    def dropEpochsByBehaviorIndices(self, bad_indices, event, keyword_in=None,
                                    keyword_out=None):
        keyword_out = keyword_in if keyword_out is None else keyword_out
        df = read_csv(self.behavior)
        epochs = self._load_epochs(event, keyword=keyword_in)
        good_indices = [i for i in range(len(df)) if i not in bad_indices]
        epochs_indices = self._behavior_to_epochs_indices(epochs, good_indices)
        self._save_epochs(epochs[epochs_indices], event, keyword=keyword_out)


    def markBadChannels(self, bad_channels, event=None, keyword_in=None,
                        keyword_out=None):
        keyword_out = keyword_in if keyword_out is None else keyword_out
        if event is None:
            raw = self._load_raw(keyword=keyword_in)
            raw.info['bads'] += bad_channels
            self._save_raw(raw, keyword=keyword_out)
        else:
            if event is 'all':
                for event in self.events:
                    self.markBadChannels(bad_channels, event=event,
                                         keyword_in=keyword_in,
                                         keyword_out=keyword_out)
            epochs = self._load_epochs(event, keyword=keyword_in)
            epochs.info['bads'] += bad_channels
            self._save_epochs(epochs, event, keyword=keyword_out)


    def alignBaselineEpochs(self, event, keyword=None):
        epochs = self._load_epochs(event, keyword=keyword)
        bl_epochs = self._load_epochs('Baseline', keyword=keyword)
        exclude = [i for i in range(len(bl_epochs)) if
                   i not in epochs.selection]
        bl_epochs.drop(exclude)
        self._save_epochs(bl_epochs, 'Baseline', keyword=keyword)


    def plotEvoked(self, event, condition=None, values=None,
                   epochs=None, keyword=None, image=True,
                   ylim={'eeg':[-10, 20]}, l_freq=None, h_freq=None,
                   tmin=None, tmax=None, detrend=1, seed=11, downsample=True,
                   picks=None, show=True):
        from mne import pick_types
        epochs = self._prepare_epochs(event, epochs, keyword, tmin, tmax,
                                      l_freq, h_freq)
        if condition is None:
            values = ['all']
            value_indices = {'all':[]}
            nTR = len(epochs)
        else:
            values = self._default_values(condition, values=values)
            value_indices = self._get_indices(epochs, condition, values)
            if downsample:
                np.random.seed(seed)
                nTR = min([len(value_indices[value]) for value in value_indices])

        if picks is not None:
           picks = pick_types(epochs.info, meg=self.meg, eeg=self.eeg,
                              eog=False, include=picks)

        x_dim = (1 + image) * (2 * self.meg + self.eeg)
        y_dim = len(values)
        fig, axs = plt.subplots(x_dim, y_dim, figsize=(5 * y_dim, 5 * x_dim))
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        if not image and self.eeg and not self.meg:
            axs = axs[np.newaxis,:]
        if not isinstance(axs, np.ndarray):
            axs = np.array([axs])
        for i, value in enumerate(values):
            if not value in value_indices:
                continue
            if value == 'all':
                evoked = epochs.average()
                indices = range(len(epochs))
            else:
                indices = value_indices[value]
                if condition is not None and downsample:
                    print('Subsampling %i/%i %s %s.' % (nTR, len(indices), condition,
                                                        value))
                    np.random.shuffle(indices)
                    indices = indices[:nTR]
                evoked = epochs[indices].average()
            if detrend:
                evoked = evoked.detrend(order=detrend)
            if y_dim > 1:
                axs2 = axs[:, i]
            else:
                axs2 = axs
            axs3 = ([axs2[0], axs2[1], axs2[2]] if self.meg and self.eeg else
                    [axs2[0]] if self.eeg else [axs2[0], axs2[1]])
            evoked.plot(axes=axs3, show=False, ylim=ylim, picks=picks)
            axs2[0].set_title('%s %s %s %s' % (self.subject, event, condition, value) +
                         (', %i trials used'% len(indices)))
            if image:
                axs3 = ([axs2[3], axs2[4], axs2[5]] if self.meg and self.eeg else
                    [axs2[1]] if self.eeg else [axs2[2], axs2[3]])
                evoked.plot_image(axes=axs3, show=False, clim=ylim, picks=picks)
        fname = self._fname('plots', 'evoked', 'jpg', keyword, event, condition,
                            *value_indices.keys())
        fig.savefig(fname)
        self._show_fig(fig, show)


    def _show_fig(self, fig, show):
        if show:
            fig.show()
        else:
            plt.close(fig)


    def _prepare_epochs(self, event, epochs, keyword, tmin, tmax,
                        l_freq, h_freq):
        tmin, tmax = self._default_t(event, tmin, tmax)
        if epochs is None:
            epochs = self._load_epochs(event, keyword=keyword)
        else:
            epochs = epochs.copy()
        if l_freq is not None or h_freq is not None:
            epochs = epochs.filter(l_freq=l_freq, h_freq=h_freq)
        epochs = epochs.crop(tmin=tmin, tmax=tmax)
        return epochs


    def _default_t(self, event, tmin=None, tmax=None, buffered=False):
        if event == 'Response':
            _, tmin2, tmax2 = self.response
        elif event == 'Baseline':
            _, tmin2, tmax2 = self.baseline
        else:
            _, tmin2, tmax2 = self.events[event]
        if tmin is None:
            tmin = tmin2
        if tmax is None:
            tmax = tmax2
        if buffered:
            tmin -= self.tbuffer
            tmax += self.tbuffer
        return tmin, tmax


    def _default_vs(self, epochs_mean, epochs_std, vmin, vmax):
        if vmin is None:
            vmin = (epochs_mean-epochs_std).min()*1.1
        if vmax is None:
            vmax = (epochs_mean+epochs_std).max()*1.1
        return vmin, vmax


    def _behavior_to_epochs_indices(self, epochs, indices):
        return [self._behavior_to_epochs_index(epochs, i) for i in indices if
                self._behavior_to_epochs_index(epochs, i)]


    def _behavior_to_epochs_index(self, epochs, ind):
        if ind in epochs.events[:, 2]:
            return list(epochs.events[:, 2]).index(ind)


    def _get_binned_indices(self, epochs, condition, bins):
        df = read_csv(self.behavior)
        bin_indices = {}
        h, edges = np.histogram([cd for cd in df[condition] if not
                                np.isnan(cd)], bins=bins)
        for j in range(1, len(edges)):
            indices = [i for i in range(len(df)) if
                       df[condition][i] >= edges[j-1] and
                       df[condition][i] <= edges[j]]
            name = '%.2f-%.2f, count %i' % (edges[j-1], edges[j], len(indices))
            bin_indices[name] = self._behavior_to_epochs_indices(epochs, indices)
        return bin_indices


    def _get_indices(self, epochs, condition, values):
        from pandas import read_csv
        df = read_csv(self.behavior)
        n = len(df)
        value_indices = {}
        if len(values) > 4 and all([isinstance(val, int) or isinstance(val, float)
                                        for val in values]):
            binsize = float(value[1] - value[0])
        for value in values:
            if len(values) > 4 and all([isinstance(val, int) or isinstance(val, float)
                                        for val in values]):
                indices = [i for i in range(n) if
                           df[condition][i] >= value - binsize/2 and
                           value + binsize/2 >= df[condition][i]]
            else:
                indices = [i for i in range(n) if df[condition][i] == value]
            epochs_indices = self._behavior_to_epochs_indices(epochs, indices)
            if epochs_indices:
                value_indices[value] = epochs_indices
        value_indices['all'] = [i for value in value_indices for i in value_indices[value]]
        return value_indices


    def channelPlot(self, event, condition, values=None, keyword=None,
                    butterfly=False, contrast=False, aux=False,
                    tmin=None, tmax=None, vmin=None, vmax=None, show=True):
        self._plotter_main(event, condition, values, butterfly=butterfly,
                           contrast=contrast, aux=aux, keyword=keyword,
                           tmin=tmin, tmax=tmax, vmin=vmin, vmax=vmax, show=show)


    def plotTFR(self, event, condition, values=None, keyword=None,
                tfr_keyword=None, power_type='npl', contrast=False,
                butterfly=False, aux=False, tmin=None, tmax=None,
                vmin=None, vmax=None, bands={'theta': (4, 8),
                'alpha': (8, 15), 'beta': (15, 35), 'low-gamma': (35, 80),
                'high-gamma': (80, 150)}):
        # computes the time frequency representation of a particular event and
        # condition or all events and conditions
        # default values are frequency from 3 to 35 Hz with 32 steps and
        # cycles from 3 to 10 s-1 with 32 steps
        tfr_keyword = keyword if tfr_keyword is None else tfr_keyword
        if bands:
            for band in bands:
                print(band + ' band')
                fmin, fmax = bands[band]
                band_struct = (band, fmin, fmax)
                self._plotter_main(event, condition, values, contrast=contrast,
                                   aux=aux, keyword=keyword, butterfly=butterfly,
                                   tfr=True, band=band_struct,
                                   tfr_keyword=tfr_keyword, power_type=power_type,
                                   tmin=tmin, tmax=tmax, vmin=vmin, vmax=vmax)
        else:
            values = self._default_values(condition, values, contrast)
            for value in values:
                self._plotter_main(event, condition,[value], contrast=contrast,
                                   aux=aux, keyword=keyword, butterfly=butterfly,
                                   tfr=True, band=None, tfr_keyword=tfr_keyword,
                                   power_type=power_type, tmin=tmin, tmax=tmax,
                                   vmin=vmin, vmax=vmax)


    def _setup_plot(self, ch_dict, butterfly=False, contrast=False, values=None):
        if butterfly:
            nplots = 1 if contrast else len(values)
            fig, ax_arr = plt.subplots(1, nplots)
            if len(values) == 1:
                ax_arr = [ax_arr]
        else:
            dim1 = int(np.ceil(np.sqrt(len(ch_dict))))
            dim2 = int(np.ceil(float(len(ch_dict))/dim1))
            fig, ax_arr = plt.subplots(dim1, dim2, sharex=True, sharey=True)
            fig.set_tight_layout(False)
            fig.subplots_adjust(left=0.05, right=0.9, top=0.9, bottom=0.1,
                                wspace=0.01, hspace=0.2)
            ax_arr = ax_arr.flatten()
            for i, ax in enumerate(ax_arr):
                ax.set_facecolor('white')
                ax.set_frame_on(False)
                if i % dim1:
                    ax.set_yticks([])
                if i < len(ax_arr)-dim2:
                    ax.set_xticks([])
        return fig, ax_arr


    def _get_ch_dict(self, inst, aux=False):
        from mne import pick_types
        if aux:
            chs = pick_types(inst.info, meg=False, eog=True, ecg=True)
        else:
            chs = pick_types(inst.info, meg=self.meg, eeg=self.eeg)
        return {ch:inst.ch_names[ch] for ch in chs}


    def _default_values(self, condition, values=None, contrast=False):
        if values is None:
            df = read_csv(self.behavior)
            values = np.unique([cd for cd in df[condition] if
                                (type(cd) is str or type(cd) is np.string_ or
                                 type(cd) is np.str_ or not np.isnan(cd))])
            if (len(values) > 5 and
                all([isinstance(val, int) or isinstance(val, float) for val in values])):
               values, edges = np.histogram(values, bins=5)
        if type(contrast) is list and len(contrast) == 2:
            values = contrast
        elif contrast:
            values = [max(values), min(values)]
        return values


    def _get_tfr_data(self, event, condition, values, keyword, power_type,
                      value_indices, band=None, mean_and_std=True,
                      band_mean=True, data_type=None):
        values_dict = {}
        frequencies_old = None
        for value in values:
            epochs_data, frequencies,_ = self._load_TFR(event, condition, value,
                                                       power_type,
                                                       data_type=data_type,
                                                       keyword=keyword)
            epochs_data = np.swapaxes(epochs_data, 2, 3)
            if frequencies_old is not None and frequencies != frequencies_old:
                raise ValueError('TFRs must be compared for the same ' +
                                 'frequencies')
            if band is not None:
                band_name, fmin, fmax = band
                band_indices = [index for index in range(len(frequencies)) if
                                frequencies[index] >= fmin and
                                frequencies[index] <= fmax]
                epochs_std = \
                    np.sqrt(epochs_data[:,:,:, band_indices].mean(axis=3)**2 +
                            epochs_data[:,:,:, band_indices].std(axis=3)**2)
                epochs_data = epochs_data[:,:,:, band_indices]
                if band_mean:
                    epochs_data = epochs_data.mean(axis=3)
            if mean_and_std:
                if band is None:
                    epochs_std = np.sqrt(epochs_data.mean(axis=0)**2+
                                         epochs_data.std(axis=0)**2)
                else:
                    epochs_std = np.sqrt(epochs_std.mean(axis=0)**2+
                                         epochs_std.std(axis=0)**2)
                epochs_mean = epochs_data.mean(axis=0)
                values_dict[value] = (epochs_mean, epochs_std)
            else:
                values_dict[value] = epochs_data
        if band is not None:
            frequencies = [f for f in frequencies if f > band[1] and f < band[2]]
        return values_dict, frequencies


    def _get_data(self, epochs, values, value_indices, mean_and_std=True):
        epochs_data = epochs.get_data()
        if mean_and_std:
            epochs_std = epochs_data.std(axis=0)
            epochs_mean = epochs_data.mean(axis=0)
            values_dict = {'all':(epochs_mean, epochs_std)}
        else:
            values_dict = {'all':epochs_data}
        for value in values:
            indices = value_indices[value]
            if mean_and_std:
                epochs_std = epochs_data[indices].std(axis=0)
                epochs_mean = epochs_data[indices].mean(axis=0)
                values_dict[value] = (epochs_mean, epochs_std)
            else:
                values_dict[value] = epochs_data[indices]
        return values_dict


    def getEventTimes(self, event):
        '''do this on the events from the raw since we don't want to have any
        unassigned events for dropped epochs in case dropped epochs need
        a designation for whatever reason'''
        raw = self._load_raw()
        stim_ch = self._get_stim_ch(event)
        events = self._find_events(raw, stim_ch)
        return raw.times[events[:, 0]]


    def _get_stim_ch(self, event):
        if event == 'Response':
            stim_ch,_,_ = self.response
        elif event == 'Baseline':
            stim_ch,_,_ = self.baseline
        else:
            stim_ch,_,_ = self.events[event]
        return stim_ch


    def _add_last_square_legend(self, fig,*labels):
        ax = fig.add_axes([0.92, 0.1, 0.05, 0.8])
        ax.axis('off')
        for label in labels:
            ax.plot(0, 0, label=label)
        ax.legend(loc='center')


    def _exclude_unpicked_types(self, inst):
        return inst.pick_types(meg=self.meg, eeg=self.eeg, eog=True,
                               ecg=True, emg=True, stim=True, ref_meg=True,
                               misc=True, resp=True, chpi=True,
                               exci=True, ias=True, syst=True, seeg=self.seeg,
                               dipole=True, gof=True, bio=True, ecog=self.ecog,
                               fnirs=True, exclude=[])


    def _pick_types(self, inst, dt, aux=False):
        return inst.copy().pick_types(meg=dt if dt in ['grad', 'mag'] else False,
                                      eeg=True if dt == 'eeg' else False,
                                      ecog=True if dt == 'ecog' else False,
                                      seeg=True if dt == 'seeg' else False,
                                      eog=aux, ecg=aux)


    def _plotter_main(self, event, condition, values, keyword=None,
                      aux=False, butterfly=False, contrast=False,
                      tfr=False, band=None, tfr_keyword=None,
                      power_type=None, tmin=None, tmax=None,
                      vmin=None, vmax=None, cpt=False, cpt_p=None,
                      cpt_keyword=None, show=True):
        heatmap = tfr and band is None
        epochs = self._load_epochs(event, keyword=keyword)
        values = self._default_values(condition, values, contrast)
        value_indices = self._get_indices(epochs, condition, values)
        tmin, tmax = self._default_t(event, tmin=tmin, tmax=tmax)
        epochs = epochs.crop(tmin=tmin, tmax=tmax)
        times = epochs.times
        for dt in self._get_data_types():
            epo = self._pick_types(epochs, dt)
            ch_dict = self._get_ch_dict(epo, aux=aux)
            fig, axs = self._setup_plot(ch_dict, butterfly=butterfly, values=values)
            if tfr:
                values_dict, frequencies = \
                    self._get_tfr_data(event, condition, values, tfr_keyword, power_type,
                                       value_indices, band=band, data_type=dt)
            else:
                values_dict = self._get_data(epo, values, value_indices)
                frequencies = None
            if contrast:
                epochs_mean0, epochs_std0 = values_dict[values[0]]
                epochs_mean1, epochs_std1 = values_dict[values[1]]
                epochs_std = np.sqrt(epochs_std0**2 + epochs_std1**2)
                epochs_mean = epochs_mean1-epochs_mean0
                if cpt:
                    clusters, cluster_p_values, times, ch_dict, frequencies, band = \
                        self._load_CPT(event, condition,
                                       '%s-%s' % (values[0], values[1]),
                                       data_type=dt, keyword=cpt_keyword)
                else:
                    clusters, cluster_p_values = None, None
                self._plot_decider(epochs_mean, epochs_std, times, axs, fig, butterfly,
                                   contrast, values, ch_dict, tfr, band, frequencies,
                                   vmin, vmax, clusters, cluster_p_values, cpt_p)
            else:
                for i, value in enumerate(values):
                    epochs_mean, epochs_std = values_dict[value]
                    if cpt:
                        clusters, cluster_p_values, times, ch_dict, frequencies, band = \
                            self._load_CPT(event, condition, value,
                                           data_type=dt,
                                           keyword=cpt_keyword)
                    else:
                        clusters, cluster_p_values = None, None
                    if butterfly:
                        axs[i].set_title(value)
                        self._plot_decider(epochs_mean, epochs_std, times, axs[i], fig,
                                           butterfly, contrast, values, ch_dict, tfr,
                                           band, frequencies, vmin, vmax, clusters,
                                           cluster_p_values, cpt_p)
                    else:
                        self._plot_decider(epochs_mean, epochs_std, times, axs, fig,
                                           butterfly, contrast, values, ch_dict, tfr,
                                           band, frequencies, vmin, vmax, clusters,
                                           cluster_p_values, cpt_p)
            if not (heatmap or butterfly):
                if contrast:
                    self._add_last_square_legend(fig, '%s-%s' % (values[0], values[1]))
                else:
                    self._add_last_square_legend(fig,*values)

            self._prepare_fig(fig, event, condition, values, aux=aux, butterfly=butterfly,
                              contrast=contrast, cpt=cpt, tfr=tfr, band=band,
                              power_type=power_type, keyword=keyword,
                              data_type=dt, show=show)


    def _plot_decider(self, epochs_mean, epochs_std, times, axs, fig, butterfly,
                      contrast, values, ch_dict, tfr, band, frequencies, vmin, vmax,
                      clusters, cluster_p_values, cpt_p):
        if epochs_mean.shape[-1] != times.shape[0] and clusters:
            raise ValueError('Time mismatch, likely you used different ' +
                             'times for the cluster permutation test')
        vmin, vmax = self._default_vs(epochs_mean[list(ch_dict.keys())],
                                     epochs_std[list(ch_dict.keys())], vmin, vmax)
        if tfr:
            if band is not None:
                self._plot_band(epochs_mean, epochs_std, times, axs, ch_dict,
                                butterfly, vmin, vmax, clusters,
                                cluster_p_values, cpt_p)
            else:
                self._plot_heatmap(epochs_mean, epochs_std, times, axs, fig,
                                   butterfly, ch_dict, frequencies, vmin, vmax,
                                   clusters, cluster_p_values, cpt_p)
        else:
            self._plot_voltage(epochs_mean, epochs_std, times, axs, butterfly,
                               ch_dict, vmin, vmax, clusters,
                               cluster_p_values, cpt_p)


    def _plot_voltage(self, epochs_mean, epochs_std, times, axs, butterfly, ch_dict,
                      vmin, vmax, clusters, cluster_p_values, cpt_p):
        for i, ch in enumerate(ch_dict):
            if butterfly:
                ax = axs
            else:
                ax = axs[i]
                ax.set_title(ch_dict[ch], fontsize=6, pad=0)
            ax.axvline(0, color='k')
            ax.set_ylim(vmin, vmax)
            v = epochs_mean[ch]-epochs_mean[ch].mean()
            lines = ax.plot(times, v)
            if not butterfly:
                ax.fill_between(times, v-epochs_std[ch], v+epochs_std[ch],
                                color=lines[0].get_color(), alpha=0.5)
        if clusters is not None and not butterfly:
            for c, p in zip(clusters, cluster_p_values):
                if p <= cpt_p:
                    for i, ch in enumerate(ch_dict):
                        sig_times = times[np.where(c[i])[0]]
                        if sig_times.size > 0:
                            h = axs[i].axvspan(sig_times[0],
                                               sig_times[-1],
                                               color='r', alpha=0.3)


    def _plot_heatmap(self, epochs_mean, epochs_std, times, axs, fig, butterfly,
                      ch_dict, frequencies, vmin, vmax, clusters,
                      cluster_p_values, cpt_p):
        from matplotlib.colors import SymLogNorm  #, LogNorm
        from matplotlib.cm import cool
        if clusters is not None:
            current_data = np.zeros(epochs_mean.shape)
            for i,(freq_cs, freq_ps) in enumerate(zip(clusters, cluster_p_values)):
                for cluster, p in zip(freq_cs, freq_ps):
                    if p <= cpt_p:
                        current_data[:,:, i] += cluster*vmax
                    else:
                        current_data[:,:, i] += cluster*vmax*0.5
        norm = SymLogNorm(vmax/10, vmin=vmin, vmax=vmax)
        tmin, tmax = times.min(), times.max()
        fmin, fmax = frequencies.min(), frequencies.max()
        extent=[tmin, tmax, fmin, fmax]
        aspect=1.0/(fmax-fmin)
        for i, ch in enumerate(ch_dict):
            if butterfly:
                ax = axs
            else:
                ax = axs[i]
                ax.set_title(ch_dict[ch], fontsize=6, pad=0)
            ax.invert_yaxis()
            if clusters is None:
                current_data = cool(norm(epochs_mean[ch].T))
                im = ax.imshow(current_data[::-1], aspect=aspect, extent=extent,
                               cmap=cool, norm=norm)
            else:
                image = np.ones((current_data.shape[0],
                                 current_data.shape[1], 3))*vmax
                image[:,:, 0] = current_data[:,:, i].T
                image[:,:, 1:2] = 0
                ax.imshow(image[::-1], aspect=aspect, norm=norm, extent=extent)
            xbuffer = (tmax-tmin)/5
            ax.set_xticks(np.round(np.linspace(tmin+xbuffer, tmax-xbuffer, 2), 2))
            frequency_labels = np.round(frequencies[::10], 2)
            ybuffer = (fmax-fmin)/5
            ax.set_yticks(np.round(np.linspace(fmin+ybuffer, fmax-ybuffer, 3), 0))
        cbar_ax = fig.add_axes([0.92, 0.1, 0.03, 0.8])
        fig.colorbar(im, cax=cbar_ax)


    def _plot_band(self, epochs_mean, epochs_std, times, axs, ch_dict, butterfly,
                   vmin, vmax, clusters, cluster_p_values, cpt_p):
        for i, ch in enumerate(ch_dict):
            if butterfly:
                ax = axs
            else:
                ax = axs[i]
                ax.set_title(ch_dict[ch], fontsize=6, pad=0)
            lines = ax.plot(times, epochs_mean[ch])
            if not butterfly:
                ax.fill_between(times, epochs_mean[ch]-epochs_std[ch],
                                epochs_mean[ch]+epochs_std[ch],
                                color=lines[0].get_color(), alpha=0.5)
            ax.axvline(0, color='k')
            ax.set_ylim(vmin, vmax)
        if clusters is not None and not butterfly:
            for c, p in zip(clusters, cluster_p_values):
                if p <= cpt_p:
                    for i, ch in enumerate(ch_dict):
                        sig_times = times[np.where(c[i])[0]]
                        if sig_times.size > 0:
                            h = axs[i].axvspan(sig_times[0],
                                               sig_times[-1],
                                               color='r', alpha=0.3)


    def _prepare_fig(self, fig, event, condition, values, aux=False,
                     butterfly=False, contrast=False, tfr=False, band=None,
                     power_type=None, cpt=False, keyword=None,
                     data_type=None, show=True):
        if tfr:
            if band:
                ylabel = 'Relative Abundance'
            else:
                ylabel = 'Frequency (Hz)'
        else:
            ylabel = r"$\mu$V"
        fig.text(0.02, 0.5, ylabel, va='center', rotation='vertical')
        fig.text(0.5, 0.02, 'Time (s)', ha='center')
        fig.set_size_inches(20, 15)
        title = (event + ' ' + condition + ' ' +
                 ' '.join([str(value) for value in values]) +
                 ' contrast'*contrast)
        if tfr and band:
            bandname,_,_ = band
            title += (' ' + bandname + ' band')
        else:
            bandname = ''
        if power_type:
            title += (' ' + {'t':'Total', 'npl':'Non-Phase-Locked',
                             'pl':'Phase-Locked'}[power_type])
        if data_type:
            title += ' ' + data_type
        fig.suptitle(title)
        fig.savefig(self._fname('plots', 'channel_plot', 'jpg',
                                contrast*'contrast', 'tfr'*tfr, 'cpt'*cpt,
                                'aux'*aux, 'butterfly'*butterfly,
                                (bandname + '_band')*(band is not None),
                                power_type, keyword, data_type,
                                event, condition,*values))
        self._show_fig(fig, show)


    def makeWavelets(self, event, condition, values=None, keyword_in=None,
                     keyword_out=None, tmin=None, tmax=None, bl_tmin=None,
                     bl_tmax=None, power_type='npl',
                     fmin=4, fmax=150, nmin=2, nmax=75, steps=32,
                     compressed=False, gain_normalize=False,
                     baseline_subtract=False, save_baseline=True,
                     overwrite=False):
        from mne.time_frequency import tfr_array_morlet
        keyword_out = keyword_in if keyword_out is None else keyword_out
        if gain_normalize and baseline_subtract:
            raise ValueError('Gain normalize and baseline subtract are ' +
                             'both True, this does not make sense')
        #note compression may not always work
        values = self._default_values(condition, values)
        frequencies = np.logspace(np.log10(fmin), np.log10(fmax), steps)
        n_cycles = np.logspace(np.log10(nmin), np.log10(nmax), steps)
        epochs = self._load_epochs(event, keyword=keyword_in)
        tmin, tmax = self._default_t(event, tmin=tmin, tmax=tmax, buffered=True)
        epochs = epochs.crop(tmin=tmin, tmax=tmax)
        value_indices = self._get_indices(epochs, condition, values)
        if gain_normalize or baseline_subtract or save_baseline:
            bl_epochs = self._load_epochs('Baseline', keyword=keyword_in)
            bl_tmin, bl_tmax = self._default_t('Baseline', tmin=bl_tmin,
                                              tmax=bl_tmax, buffered=True)
            bl_epochs = bl_epochs.crop(tmin=bl_tmin, tmax=bl_tmax)
            bl_value_indices = self._get_indices(bl_epochs, condition, values)
        for dt in self._get_data_types():
            print(dt)
            epo = self._pick_types(epochs, dt)
            values_dict = self._get_data(epo, values, value_indices,
                                         mean_and_std=False)
            if gain_normalize or baseline_subtract or save_baseline:
                bl_epo = self._pick_types(bl_epochs, dt)
                bl_values_dict = self._get_data(bl_epo, values, bl_value_indices,
                                                mean_and_std=False)
            for value in values:
                if (self._has_TFR(event, condition, value, power_type,
                                  data_type=dt, keyword=keyword_out) and
                    not overwrite):
                    self._overwrite_error('TFR', power_type, keyword=keyword_out)
                if gain_normalize or baseline_subtract or save_baseline:
                    if power_type == 't':
                        bl_current_data = bl_values_dict[value]
                    elif power_type == 'npl':
                        bl_current_data = (bl_values_dict[value] -
                                           bl_values_dict[value].mean(axis=0))
                    elif power_type == 'pl':
                        bl_current_data = bl_values_dict[value].mean(axis=0)
                        bl_current_data = np.expand_dims(bl_current_data, axis=0)
                    else:
                        raise ValueError('Unrecognized power type %s ' % power_type +
                                         'please use \'t\', \'npl\' or \'pl\'')
                    bl_tfr = tfr_array_morlet(bl_current_data,
                                              sfreq=bl_epochs.info['sfreq'],
                                              freqs=frequencies, n_cycles=n_cycles,
                                              output='power')
                    bl_epo = bl_epo.crop(tmin=bl_tmin+self.tbuffer,
                                         tmax=bl_tmax-self.tbuffer)
                    bl_tind = bl_epochs.time_as_index(bl_epo.times) #crop buffer
                    bl_tfr = np.take(bl_tfr, bl_tind, axis=-1)
                    if save_baseline:
                        self._save_TFR(bl_tfr, frequencies, n_cycles, 'Baseline',
                                       condition, value, power_type,
                                       data_type=dt, keyword=keyword_out,
                                       compressed=compressed)
                    bl_power = bl_tfr.mean(axis=0).mean(axis=-1) #average over epochs, times
                    bl_power = bl_power[np.newaxis,:,:, np.newaxis]
                if power_type == 't':
                    current_data = values_dict[value]
                elif power_type == 'npl':
                    current_data = (values_dict[value] - 
                                    values_dict[value].mean(axis=0))
                elif power_type == 'pl':
                    current_data = values_dict[value].mean(axis=0)
                    current_data = np.expand_dims(current_data, axis=0)
                else:
                    raise ValueError('Unrecognized power type %s ' % power_type +
                                     'please use \'t\', \'npl\' or \'pl\'')
                tfr = tfr_array_morlet(current_data, sfreq=epochs.info['sfreq'],
                                       freqs=frequencies, n_cycles=n_cycles,
                                       output='power')
                epo = epo.crop(tmin=tmin+self.tbuffer, tmax=tmax-self.tbuffer)
                tind = epochs.time_as_index(epo.times)
                tfr = np.take(tfr, tind, axis=-1)
                if gain_normalize or baseline_subtract:
                    tile_shape = (tfr.shape[0], 1, 1, tfr.shape[-1])
                    bl_power = np.tile(bl_power, tile_shape)
                    if gain_normalize:
                        tfr /= bl_power #normalize by gain compared to baseline
                    if baseline_subtract:
                        tfr -= bl_power #subtract baseline power to normalize
                self._save_TFR(tfr, frequencies, n_cycles, event, condition, value,
                               power_type, data_type=dt, keyword=keyword_out,
                               compressed=compressed)
                del tfr
                del bl_tfr


    def psdMultitaper(self, keyword=None, ch='Oz', N=6, deltaN=0.25, NW=3.0,
                      fmin=0.5, fmax=25, BW=1.0, assign_states=True,
                      labels={'Wake':'red', 'Sleep':'white'}, overwrite=False,
                      n_jobs=10, vmin=None, vmax=None, adaptive=False,
                      jackknife=True, low_bias=True):
        # full-night: N = 30.0 s, deltaN = 5.0 s, deltaf = 1.0 Hz, TW = 15, L = 29
        # ultradian: N = 6.0 s, deltaN = 0.25 s, deltaf = 1.0 Hz, TW = 3, L = 5
        # microevent: N = 2.5 s, deltaN = 0.05 s, deltaf = 4.0 Hz, TW = 5, L = 9
        #ch = 'EEG072'
        try:
            import nitime.algorithms as tsa
        except:
            print('Unable to import nitime... you won\'t be able to use this feature')
        from .psd_multitaper_plot_tools import ButtonClickProcessor
        from joblib import Parallel, delayed
        raw = self._load_raw(keyword=keyword)

        ch_ind = raw.ch_names.index(ch)
        raw_data = raw.get_data(picks=ch_ind).flatten()
        Fs = raw.info['sfreq']

        n_full_windows = int(np.floor(raw.times[-1]/N))
        t_end = raw.times[int(n_full_windows*N*Fs)]
        n_windows = int((n_full_windows-1) * (N/deltaN)) + 1

        if self._has_PSD(keyword) and not overwrite:
            image = self._load_PSD(keyword)
        else:
            imsize = int(Fs/2*N) + 1
            image = np.zeros((imsize, int(n_full_windows*(N/deltaN))))
            counters = np.zeros((int(n_full_windows*(N/deltaN))))
            with Parallel(n_jobs=n_jobs) as parallel:
                results = parallel(delayed(tsa.multi_taper_psd)(
                            raw_data[int(i*deltaN*Fs):int(i*deltaN*Fs)+int(N*Fs)],
                            Fs=Fs, NW=NW, BW=BW, adaptive=adaptive,
                            jackknife=jackknife, low_bias=low_bias)
                                    for i in tqdm(range(n_windows)))
            fs, psd_mts, nus = zip(*results)
            for i in range(n_windows):
                for j in range(i, i+int(N/deltaN)):
                    image[:, j] += np.log10(psd_mts[i])
                    counters[j] += 1
            for k in range(imsize):
                image[k] /= counters
            f = np.linspace(0, Fs/2, imsize)
            f_inds = [i for i, freq in enumerate(f) if
                      (freq >= fmin and freq <= fmax)]
            image = image[f_inds]
            self._save_PSD(image, keyword)

        fig2, ax2 = plt.subplots()
        fig2.set_size_inches(12, 8)
        fig2.subplots_adjust(right=0.8)
        im2 = ax2.imshow(image, aspect='auto', cmap='jet', vmin=vmin, vmax=vmax)

        cax2 = fig2.add_axes([0.82, 0.1, 0.05, 0.8])
        fig2.colorbar(im2, cax=cax2)

        fig2.suptitle('Multitaper Spectrogram')

        if assign_states:
            drs = {label:[] for label in labels}
            fig, ax1 = plt.subplots()
            fig.suptitle('Multitaper Spectrogram')
            fig.set_size_inches(12, 8)
            fig.subplots_adjust(right=0.7)
            buttons = []
            button_height = 0.8/(len(labels)+1)
            y0 = 0.1 + button_height/(len(labels)+1)
            for label in labels:
                label_ax = fig.add_axes([0.85, y0, 0.1, button_height])
                y0 += button_height + button_height/(len(labels)+1)
                buttons.append(ButtonClickProcessor(label_ax, label, labels[label],
                                                    ax1, drs, image))
            im = ax1.imshow(image, aspect='auto', cmap='jet', vmin=vmin, vmax=vmax)
            cax = fig.add_axes([0.72, 0.1, 0.05, 0.8])
            fig.colorbar(im, cax=cax)
            axs = [ax1, ax2]
        else:
            axs = [ax2]

        for ax in axs:
            ax.invert_yaxis()
            ax.set_yticks(np.linspace(0, image.shape[0], 10))
            ax.set_yticklabels(np.round(np.linspace(fmin, fmax, 10), 1))
            ax.set_ylabel('Frequency (Hz)')
            ax.set_xticks(np.linspace(0, image.shape[1], 10))
            ax.set_xticklabels(np.round(np.linspace(0, t_end, 10)))
            ax.set_xlabel('Time (s)')
        fig2.savefig(self._fname('plots', 'psd_multitaper', 'jpg',
                                 'N_%i_dN_%.2f' % (N, deltaN),
                                 'fmin_%.2f_fmax_%.2f_NW_%i' % (fmin, fmax, NW)))
        plt.close(fig2)

        if assign_states:
            plt.show(fig)
            state_times = {label:[] for label in labels}
            for label in drs:
                for dr in drs[label]:
                    rect = dr.rect
                    start = rect.get_x()*deltaN
                    duration = rect.get_width()*deltaN
                    state_times[label].append((start,(start+duration)))
            return state_times


    def assignConditionFromStateTimes(self, event, state_times,
                                      condition='State', no_state='Neither'):
        event_times = self.getEventTimes(event)
        states = np.tile(no_state, len(event_times))
        for i, t in enumerate(event_times):
            for state in state_times:
                if any([t <= tmax and t >= tmin for
                        tmin, tmax in state_times[state]]):
                    states[i] = state
        try:
            df = read_csv(self.behavior)
            df[condition] = states
        except Exception as e:
            df = DataFrame({condition:states})
            self._add_meta_data({'Behavior':self.behavior}, overwrite=True)
        if not op.isdir(op.dirname(self.behavior)):
            os.makedirs(op.dirname(self.behavior))
        df.to_csv(self.behavior)


    def CPTByBand(self, event, condition, values=None, keyword_in=None,
                  keyword_out=None, tfr_keyword=None, power_type='npl',
                  tmin=None, tmax=None, threshold=6.0, aux=False,
                  bands={'theta':(4, 8), 'alpha':(8, 15),
                         'beta':(15, 35), 'low-gamma':(35, 80),
                         'high-gamma':(80, 150)},
                  contrast=False):
        for band in bands:
            fmin, fmax = bands[band]
            band_struct = (band, fmin, fmax)
            self.CPT(event, condition, values, keyword_in=keyword_in,
                     keyword_out=keyword_out, tfr_keyword=tfr_keyword,
                     power_type=power_type, tmin=tmin, tmax=tmax,
                     tfr=True, threshold=threshold, band=band_struct,
                     contrast=contrast, aux=aux)


    def CPT(self, event, condition, values=None, keyword_in=None,
            keyword_out=None, tmin=None, tmax=None, bl_tmin=None, bl_tmax=None,
            aux=False, alpha=0.3, threshold=6.0, tfr=False, band=None,
            tfr_keyword=None, power_type='npl', contrast=False,
            n_permutations=2000, n_jobs=10, overwrite=False):
        keyword_out = keyword_in if keyword_out is None else keyword_out
        tfr_keyword = keyword_in if tfr_keyword is None else tfr_keyword
        if power_type == 'pl':
            raise ValueError('No epochs to permute for phase-locked power')
        values = self._default_values(condition, values, contrast)
        if all([self._has_CPT(event, condition, value=value, data_type=dt,
                              keyword=keyword_out) for dt in self._get_data_types()
                              for value in values]) and not overwrite:
            self._overwrite_error('CPT', event=event, condition=condition,
                                  keyword=keyword_out)
        tmin, tmax = self._default_t(event, tmin, tmax)
        epochs = self._load_epochs(event, keyword=keyword_in)
        epochs = epochs.crop(tmin=tmin, tmax=tmax)
        value_indices = self._get_indices(epochs, condition, values)
        if not contrast:
            bl_epochs = self._load_epochs('Baseline', keyword=keyword_in)
            bl_tmin, bl_tmax = self._default_t('Baseline', tmin=bl_tmin,
                                              tmax=bl_tmax)
            bl_epochs = bl_epochs.crop(tmin=bl_tmin, tmax=bl_tmax)
            bl_value_indices = self._get_indices(bl_epochs, condition, values)
        for dt in self._get_data_types():
            epo = self._pick_types(epochs, dt)
            ch_dict = self._get_ch_dict(epo, aux=aux)
            if not contrast:
                bl_epo = self._pick_types(bl_epochs, dt)
            if tfr:
                values_dict, frequencies = \
                    self._get_tfr_data(event, condition, values,
                                       tfr_keyword, power_type, value_indices,
                                       band=band, mean_and_std=False,
                                       data_type=dt)
                if not contrast:
                    bl_values_dict, bl_frequencies = \
                        self._get_tfr_data('Baseline', condition, values,
                                           tfr_keyword, power_type,
                                           bl_value_indices,
                                           band=band, mean_and_std=False,
                                           data_type=dt)
            else:
                values_dict = self._get_data(epo, values,
                                             value_indices, mean_and_std=False)
                if not contrast:
                    bl_values_dict = self._get_data(bl_epo, values,
                                                    bl_value_indices,
                                                    mean_and_std=False)
                frequencies = None
            if contrast:
                epochs_data0 = values_dict[values[0]]
                epochs_data1 = values_dict[values[1]]
                clusters, cluster_p_values = \
                        self._CPT(epochs_data0, epochs_data1, threshold,
                                  n_permutations=n_permutations, n_jobs=n_jobs)
                self._save_CPT(clusters, cluster_p_values, epochs.times, ch_dict,
                               frequencies, band, event, condition,
                               '%s-%s' % (values[0], values[1]),
                               data_type=dt, keyword=keyword_out)
            else:
                for value in values:
                    evo_data = values_dict[value]
                    bl_evo_data = bl_values_dict[value]
                    bl_evo_data = self._equalize_baseline_length(evo_data, bl_evo_data)
                    clusters, cluster_p_values = \
                        self._CPT(evo_data, bl_evo_data, threshold,
                                  n_permutations=n_permutations, n_jobs=n_jobs)
                    self._save_CPT(clusters, cluster_p_values, epochs.times, ch_dict,
                                   frequencies, band, event, condition,
                                   value, data_type=dt, keyword=keyword_out)


    def _equalize_baseline_length(self, value_data, bl_data):
        bl_len = bl_data.shape[2]
        val_len = value_data.shape[2]
        if bl_len < val_len:
            n_reps = int(val_len / bl_len)
            remainder = val_len % bl_len
            print('Using %.2f ' % (n_reps + float(remainder)/bl_len) +
                  'repetitions of the baseline period for permuation')
            baseline = np.tile(bl_data, n_reps)
            remainder_baseline = np.take(bl_data,
                                         range(bl_len-remainder, bl_len),
                                         axis=2)
                              #take from the end of the baseline period
            bl_data = np.concatenate((baseline, remainder_baseline), axis=2)
        elif bl_len > val_len:
            bl_data = np.take(bl_data, range(val_len), axis=2)
        return bl_data


    def _CPT(self, data0, data1, threshold, n_permutations=1000, n_jobs=10):
        from mne.stats import permutation_cluster_test
        if data0.shape != data1.shape:
            raise ValueError('Cluster Permutation size mismatch error')
        if data0.ndim > 4:
            raise ValueError('Too many data dimensions')
        elif data0.ndim == 4: 
            clusters, cluster_p_values = [],[]
            for i in tqdm(range(data0.shape[3])):
                T_obs, this_clusters, this_cluster_p_values, H0 = \
                    permutation_cluster_test([data0[:, :, :, i], data1[:, :, :, i]],
                                             n_permutations=n_permutations,
                                             threshold=threshold,
                                             tail=0 if threshold else None,
                                             n_jobs=n_jobs, buffer_size=None,
                                             verbose=False)
                clusters.append(this_clusters)
                cluster_p_values.append(this_cluster_p_values)
        else:
            T_obs, clusters, cluster_p_values, H0 = \
                permutation_cluster_test([data0, data1],
                                         n_permutations=n_permutations,
                                         threshold=threshold,
                                         tail=0 if threshold else None,
                                         n_jobs=n_jobs, buffer_size=None,
                                         verbose=False)
        return clusters, cluster_p_values


    def plotCPT(self, event, condition, values=None, keyword_in=None,
                keyword_out=None, aux=False, vmin=None, vmax=None,
                contrast=False, cpt_p=0.01, show=True):
        keyword_out = keyword_in if keyword_out is None else keyword_out
        values = self._default_values(condition, values, contrast)
        for value in values:
            self._plotter_main(event, condition,[value], aux=aux,
                               contrast=contrast, keyword=keyword_in,
                               cpt=True, cpt_keyword=keyword_out, cpt_p=cpt_p,
                               vmin=vmin, vmax=vmax, show=show)


    def plotCPTTFR(self, event, condition, values=None, keyword_in=None,
                   keyword_out=None, aux=False, vmin=None, vmax=None,
                   tfr_keyword=None, power_type='npl',
                   bands={'theta':(4, 8), 'alpha':(8, 15),
                         'beta':(15, 35), 'low-gamma':(35, 80),
                         'high-gamma':(80, 150)},
                   cpt_p=0.01, show=True):
        tfr_keyword = keyword if tfr_keyword is None else tfr_keyword
        if bands:
            for band in bands:
                print(band + ' band')
                fmin, fmax = bands[band]
                band_struct = (band, fmin, fmax)
                self._plotter_main(event, condition, values, contrast=contrast,
                                   aux=aux, keyword=keyword_in, butterfly=butterfly,
                                   tfr=True, band=band_struct,
                                   tfr_keyword=tfr_keyword, power_type=power_type,
                                   cpt=True, cpt_keyword=keyword_out, cpt_p=cpt_p,
                                   tmin=tmin, tmax=tmax, vmin=vmin, vmax=vmax)
        else:
            values = self._default_values(condition, values, contrast)
            for value in values:
                self._plotter_main(event, condition,[value], contrast=contrast,
                                   aux=aux, keyword=keyword_in, butterfly=butterfly,
                                   tfr=True, band=None, tfr_keyword=tfr_keyword,
                                   power_type=power_type, cpt=True, cpt_p=cpt_p,
                                   cpt_keyword=keyword_out, tmin=tmin, tmax=tmax,
                                   vmin=vmin, vmax=vmax)


    def plotControlVariables(self, conditions=None, show=True):
        import seaborn as sns
        df = read_csv(self.behavior)
        n = len(df)
        if conditions is None:
            conditions = [param for param in list(df) if
                          not 'unnamed' in param.lower()]
        for param in conditions:
            fig,(ax1, ax2) = plt.subplots(1, 2)
            if any([isinstance(df[param][index], str)
                    for index in range(n)]):
                sns.countplot(df[param], ax=ax1)
                sns.swarmplot(x=range(n), y=df[param], ax=ax2)
            else:
                var = [v for v in df[param] if not np.isnan(v)]
                t = [i for i, v in enumerate(df[param]) if not
                        np.isnan(v)]
                sns.distplot(var, bins=10, ax=ax1)
                ax2 = sns.pointplot(x=t, y=var, join=False, ax=ax2)
                xticks = np.linspace(min(t), max(t), 10)
                ax2.set_xticks(xticks)
                ax2.set_xticklabels(['%.1f' % tick for tick in xticks])
            ax1.set_title('Histogram')
            ax1.set_xlabel(param)
            ax2.set_title('Time Course')
            ax2.set_xlabel('Time')
            ax2.set_ylabel(param)
            fig.savefig(self._fname('plots', 'var', 'png', param))
            self._show_fig(fig, show)

        params = conditions
        for i in range(len(conditions)):
            for j in range(i+1, len(conditions)):
                fig, ax = plt.subplots()
                iscat1 = (any([isinstance(df[params[i]][index], str)
                              for index in range(n)]) or
                          len(np.unique(df[params[i]]))>5)
                iscat2 = (any([isinstance(df[params[j]][index], str)
                              for index in range(n)]) or
                          len(np.unique(df[params[j]]))>5)
                if iscat1 and iscat2:
                    sns.countplot(x=df[params[i]], hue=df[params[j]], ax=ax)
                elif iscat1 or iscat2:
                    sns.stripplot(x=df[params[i]], y=df[params[j]], jitter=True)
                    sns.violinplot(x=df[params[i]], y=df[params[j]])
                else:
                    indices = [index for index in range(n) if not
                               (np.isnan(df[params[i]][index]) or
                                np.isnan(df[params[j]][index]))]
                    column1 = [df[params[i]][index]
                               for index in indices]
                    column2 = [df[params[j]][index]
                               for index in indices]
                    heatmat = np.histogram2d(column1, column2, bins=9)
                    centers1 = (heatmat[1][:-1] +
                                float(heatmat[1][1]-heatmat[1][0])/2)
                    centers1 = [round(c, 2) for c in centers1]
                    centers2 = (heatmat[2][:-1] +
                                float(heatmat[2][1]-heatmat[2][0])/2)
                    centers2 = [round(c, 2) for c in centers2]
                    heatdf = DataFrame(columns=centers1, index=centers2,
                                       data=heatmat[0])
                    sns.heatmap(heatdf, ax=ax)
                    ax.invert_yaxis()
                ax.set_title('%s %s distribution' % (params[i], params[j]))
                ax.set_xlabel(params[i])
                ax.set_ylabel(params[j])
                if (len(np.unique(df[params[i]])) > 10 and
                    all([isinstance(p, (int, float, complex))
                         for p in df[params[i]]])):
                    xticks = np.linspace(min(df[params[i]]), max(df[params[i]]), 10)
                    ax.set_xticks(xticks)
                    ax.set_xticklabels(['%.1f' % tick for tick in xticks])
                if not (iscat1 and iscat2):
                    if (len(np.unique(df[params[j]])) > 10 and
                        all([isinstance(p, (int, float, complex))
                             for p in df[params[j]]])):
                        yticks = np.linspace(min(df[params[j]]), max(df[params[j]]), 10)
                        ax.set_yticks(yticks)
                        ax.set_yticklabels(['%.1f' % tick for tick in yticks])
                fig.savefig(self._fname('plots', 'comparison', 'png', params[i],
                            params[j]))
                self._show_fig(fig, show)


    def markAutoReject(self, event, keyword_in=None, keyword_out=None,
                       bad_ar_threshold=0.5, n_jobs=10,
                       n_interpolates=[1, 2, 3, 5, 7, 10, 20], random_state=89,
                       consensus_percs=np.linspace(0, 1.0, 11), overwrite=False):
        try:
            from autoreject import AutoReject
        except:
            print('Unable to import autoreject... you won\'t be able to use this feature ' +
                  'unless you install autoreject and its sklearn dependancy')
        from mne import pick_types
        keyword_out = keyword_in if keyword_out is None else keyword_out
        if self._has_epochs(event, keyword_out) and not overwrite:
           self._overwrite_error('Epochs', event=event, keyword=keyword_out)
        epochs = self._load_epochs(event, keyword=keyword_in)
        picks = pick_types(epochs.info, meg=self.meg, eeg=self.eeg, stim=False,
                           eog=False, exclude=epochs.info['bads'])
        ar = AutoReject(n_interpolates, consensus_percs,
                        picks=picks, random_state=random_state,
                        n_jobs=n_jobs, verbose='tqdm')
        epochs_ar, reject_log = ar.fit_transform(epochs, return_log=True)
        rejected = float(sum(reject_log.bad_epochs))/len(epochs)
        print('\n\n\n\n\n\nAutoreject rejected %.0f%% of epochs\n\n\n\n\n\n' % (100*rejected))
        self._save_epochs(epochs_ar, event, keyword=keyword_out)
        self._save_autoreject(ar, reject_log, event, keyword=keyword_out)


    def plotAutoReject(self, event, keyword_in=None, keyword_out=None,
                       ylim=dict(eeg=(-30, 30)), show=True):
        try:
            from autoreject import set_matplotlib_defaults
        except:
            print('Unable to import autoreject... you won\'t be able to use this feature ' +
                  'unless you install autoreject and its sklearn dependancy')
        import matplotlib.patches as patches
        from matplotlib.cm import viridis
        keyword_out = keyword_in if keyword_out is None else keyword_out
        epochs_ar = self._load_epochs(event, keyword=keyword_out)
        epochs_comparison = self._load_epochs(event, keyword=keyword_in)
        ar, reject_log = self._load_autoreject(event, keyword=keyword_out)

        set_matplotlib_defaults(plt, style='seaborn-white')
        if self.eeg:
            losses = {'eeg':ar.loss_['eeg'].mean(axis=-1)}  # losses are stored by channel type.
        else:
            losses = {'grad':ar.loss_['grad'].mean(axis=-1),
                    'mag':ar.loss_['mag'].mean(axis=-1)}
                      # losses are stored by channel type.
        fig = epochs_ar.plot_drop_log(show=False)
        self._show_fig(fig, show)

        for modality in losses:
            loss = losses[modality]
            fig, ax = plt.subplots()
            im = ax.matshow(loss.T * 1e6, cmap=viridis)
            ax.set_xticks(range(len(ar.consensus)))
            ax.set_xticklabels(ar.consensus)
            ax.set_yticks(range(len(ar.n_interpolate)))
            ax.set_yticklabels(ar.n_interpolate)

            # Draw rectangle at location of best parameters
            idx, jdx = np.unravel_index(loss.argmin(), loss.shape)
            rect = patches.Rectangle((idx - 0.5, jdx - 0.5), 1, 1, linewidth=2,
                                     edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.xaxis.set_ticks_position('bottom')
            ax.set_xlabel(r'Consensus percentage $\kappa$')
            ax.set_ylabel(r'Max sensors interpolated $\rho$')
            ax.set_title('Mean cross validation error (x 1e6) %s' % modality)
            fig.colorbar(im)
            fig.savefig(self._fname('plots', 'ar', 'jpg', event, modality,
                                    'consensus_v_interpolates'))
            self._show_fig(fig, show)

        fig = epochs_comparison.average().plot(ylim=ylim, spatial_colors=True,
                                               window_title='Before Autoreject',
                                               show=False)
        fig.savefig(self._fname('plots', 'ar', 'jpg', event,
                                'ar_before'))
        self._show_fig(fig, show)
        epochs_ar.average().plot(ylim=ylim, spatial_colors=True,
                                 window_title='After Autoreject', show=False)
        fig.savefig(self._fname('plots', 'ar', 'jpg', event,
                                'ar_after'))
        self._show_fig(fig, show)


    def plotRejectLog(self, event, keyword_in=None, keyword_out=None,
                      scalings=dict(eeg=40e-6)):
        ar, reject_log = self._load_autoreject(event, keyword=keyword_out)
        epochs_comparison = self._load_epochs(event, keyword=keyword_in)
        reject_log.plot_epochs(epochs_comparison, scalings=scalings,
                               title='Dropped and interpolated Epochs')


    def epochs2source(self, event, condition, values=None, snr=1.0,
                      keyword_in=None, keyword_out=None, method='dSPM',
                      pick_ori='normal', shared_baseline=False,
                      save_baseline_stc=False, overwrite=False):
        from mne.minimum_norm import apply_inverse
        if not all([self.fs_subjects_dir, self.bemf, self.srcf, self.transf]):
            raise ValueError('Source estimation parameters not defined, ' +
                             'either add to meta data or redefine MEEGbuddy')

        values = self._default_values(condition, values=values)
        keyword_out = keyword_in if keyword_out is None else keyword_out

        if (all([self._has_source(event, condition, value, keyword_out)
                 for value in values]) and not overwrite):
            self._overwrite_error('Source Estimate', event=event,
                                  condition=condition, values=values,
                                  keyword=keyword_out)

        epochs = self._load_epochs(event, keyword=keyword_in)
        bl_epochs = self._load_epochs('Baseline', keyword=keyword_in)
        value_indices = self._get_indices(epochs, condition, values)

        bem, source, coord_trans, lambda2, epochs, bl_epochs, fwd = \
            self._source_setup(event, snr, epochs, bl_epochs)

        print('Making inverse for all...')
        inv = self._generate_inverse(epochs, fwd, bl_epochs, lambda2, method,
                                     pick_ori)
        self._save_inverse(inv, lambda2, method, pick_ori,
                           event, condition, 'all', keyword_out)
        if save_baseline_stc:
            print('Applying inverse on baseline for all...')
            bl_evoked = bl_epochs.average()
            bl_stc = apply_inverse(bl_evoked, inv, lambda2=lambda2, method=method,
                                   pick_ori=pick_ori)
            self._save_source(bl_stc, 'Baseline', condition, 'all', keyword_out)

        if not shared_baseline:
            bl_value_indices = self._get_indices(bl_epochs, condition, values)
        for value in values:
            if not shared_baseline:
                if not value in bl_value_indices: #if the baseline was corrupted,
                    continue                      #don't use the trial
            bl_indices = bl_value_indices[value]
            print('Making inverse for %s...' % value)
            inv = self._generate_inverse(epochs, fwd, bl_epochs[bl_indices],
                                         lambda2, method, pick_ori)
            self._save_inverse(inv, lambda2, method, pick_ori,
                               event, condition, value, keyword_out)
            if save_baseline_stc:
                print('Applying inverse on baseline for %s...' % value)
                bl_evoked = bl_epochs[bl_indices].average()
                bl_stc = apply_inverse(bl_evoked, inv, lambda2=lambda2,
                                       method=method, pick_ori=pick_ori)
                self._save_source(bl_stc, 'Baseline', condition, value,
                                  keyword_out)
            print('Applying inverse for %s' % value)
            indices = value_indices[value]
            current_epochs = epochs[indices]
            evoked = current_epochs.average()
            stc = apply_inverse(evoked, inv, lambda2=lambda2, method=method,
                                pick_ori=pick_ori)
            self._save_source(stc, event, condition, value, keyword_out)


    def _source_setup(self, event, snr, epochs, bl_epochs):
        from mne.utils import set_config
        from mne import read_trans, read_bem_solution, make_forward_solution, read_source_spaces
        set_config("SUBJECTS_DIR", self.fs_subjects_dir, set_env=True)

        bem = read_bem_solution(self.bemf)
        source = read_source_spaces(self.srcf)
        coord_trans = read_trans(self.transf)

        # Source localization parameters.
        lambda2 = 1.0 / snr ** 2

        tmin, tmax = self._default_t(event, tmin=None, tmax=None, buffered=True)
        epochs = epochs.crop(tmin=tmin, tmax=tmax)
        if self.eeg:
            epochs = epochs.set_eeg_reference(ref_channels='average',
                                              projection=True)

        bads = list(np.unique(epochs.info['bads'] + bl_epochs.info['bads']))
        epochs.info['bads'] = bads
        bl_epochs.info['bads'] = bads

        _, bl_tmin, bl_tmax = self.baseline
        bl_epochs = bl_epochs.crop(tmin=bl_tmin, tmax=bl_tmax)
        if self.eeg:
            bl_epochs = bl_epochs.set_eeg_reference(ref_channels='average',
                                                    projection=True)
        print('Making forward model...')
        fwd = make_forward_solution(epochs.info, coord_trans, source, bem,
                                    meg=self.meg, eeg=self.eeg, mindist=1.0)
        return bem, source, coord_trans, lambda2, epochs, bl_epochs, fwd


    def _generate_inverse(self, epochs, fwd, bl_epochs, lambda2, method, pick_ori):
        from mne.minimum_norm import make_inverse_operator
        from mne import compute_covariance
        noise_cov = compute_covariance(bl_epochs, method="shrunk")
        inv = make_inverse_operator(epochs.info, fwd, noise_cov)
        return inv


    def fsaverageMorph(self, event, condition, values=None, keyword_in=None,
                       keyword_out=None, overwrite=False):
        keyword_out = keyword_in if keyword_out is None else keyword_out
        values = self._default_values(condition, values=values)
        for value in values:
            stc = self._load_source(event, condition, value, keyword=keyword_in)
            stc_fs = stc.morph('fsaverage')
            if overwrite or not self._has_source(event, condition, value,
                                                 keyword=keyword_out):
                self._save_source(stc_fs, event, condition, value,
                                  keyword=keyword_out)


    def sourceContrast(self, event, condition, values=None, keyword=None):
        if len(values) != 2:
            raise ValueError('Can only contrast two values at once')
        else:
            stc0 = self._load_source(event, condition, values[0])
            stc1 = self._load_source(event, condition, values[1])
            if stc0 and stc1:
                stc_con = stc0-stc1
                self._save_source(stc_con, event, condition,
                                  '%s %s Contrast' % (values[0], values[1]),
                                  keyword=keyword)


    def plotSourceSpace(self, event, condition, values=None,
                        tmin=None, tmax=None, keyword=None, downsample=False,
                        seed=11, hemi='both', size=(800, 800), time_dilation=25,
                        fps=20, clim='auto', use_saved_stc=False, gif_combine=True,
                        views=['lat', 'med', 'cau', 'dor', 'ven', 'fro', 'par'],
                        show=True):
        ''' This may take some time... plots each individual view and then
            combines them all in one animated gif is gif_combine=True. It's okay
            to have other windows over the mlab plots but don't minimize them
            otherwise the image write out will break! '''
        try:
            from mayavi import mlab
        except:
            print('Unable to import mayavi')
        from .gif_combine import combine_gifs
        from mne.minimum_norm import apply_inverse
        values = self._default_values(condition, values=values)
        tmin, tmax = self._default_t(event, tmin, tmax)
        if not use_saved_stc:
            epochs = self._load_epochs(event, keyword=keyword)
            epochs = epochs.crop(tmin=tmin, tmax=tmax)
            if self.eeg:
                epochs = epochs.set_eeg_reference(ref_channels='average',
                                                  projection=True, verbose=False)
            value_indices = self._get_indices(epochs, condition, values)
            nTR = min([len(value_indices[value]) for value in value_indices])
            if downsample:
                np.random.seed(seed)
        for value in values:
            if use_saved_stc:
                if not self._has_source(event, condition, value, keyword=keyword):
                    self._no_file_error('Source Estimate', event=event,
                                        condition=condition, value=value,
                                        keyword=keyword)
                stc = self._load_source(event, condition, value,
                                        keyword=keyword)
                stc.crop(tmin=tmin, tmax=tmax)
            else:
                indices = value_indices[value]
                if downsample:
                    print('Subsampling %i/%i ' % (nTR, len(indices)) +
                          'for %s %s.' % (condition, value))
                    np.random.shuffle(indices)
                    indices = sorted(indices[:nTR])
                inv, lambda2, method, pick_ori = \
                    self._load_inverse(event, condition, value, keyword=keyword)
                stc_Evoked = epochs[indices].average()
                stc = apply_inverse(stc_Evoked, inv, lambda2=lambda2,
                                    method=method, pick_ori=pick_ori, verbose=False)
            #if clim == 'default':
            #    clim = dict(kind='value', lims=(stc.data.min(), stc.data.mean(),
            #                                   stc.data.max()))

            gif_names = []
            for view in views:
                gif_name = self._fname('plots', 'source_plot', 'gif', event,
                                       condition, value, keyword, hemi, view)
                fig = [mlab.figure(size=size)]
                fig = stc.plot(subjects_dir=self.fs_subjects_dir,
                               subject=self.subject, hemi=hemi, clim=clim,
                               views=[view], figure=fig, colormap='mne')
                fig.save_movie(gif_name, time_dilation=time_dilation)
                fig.close()
                gif_names.append(gif_name)

            if gif_combine:
                print('Combining gifs for %s' % value)
                anim = combine_gifs(self._fname('plots', 'source_plot', 'gif',
                                                event, condition, value,
                                                keyword, hemi, *views),
                                    fps, *gif_names)
                if show:
                    plt.show()
            plt.close('all')


    def interpolateArtifact(self, event, use_raw=True, keyword=None, mode='spline',
                            npoint_art=1, offset=0, points=10, k=3):
        #note: points and k not used in mne's interpolation (linear and window)
        #note: linear and window methods interpolate auxillary channels via spline
        from mne import EpochsArray, pick_types, BaseEpochs
        from mne.io import RawArray
        from mne.preprocessing import fix_stim_artifact
        if mode not in ('spline', 'linear', 'window'):
            raise ValueError('Mode must be spline/linear/window')
        stim_ch,_,_ = self.events[event]
        if use_raw:
            inst = self._load_raw(keyword=keyword)
            events = self._find_events(inst, stim_ch)
        else:
            inst = self._load_epochs(event, keyword=keyword)
            events = None
        if mode in ('linear', 'window'):
            sfreq = inst.info['sfreq']
            tmin = -float(offset)/sfreq
            tmax = float(npoint_art-offset)/sfreq
            print('Interpolating using tmin %.3f tmax %.3f' % (tmin, tmax))
            interp = fix_stim_artifact(inst.copy(), events=events, tmin=tmin,
                                       tmax=tmax, mode=mode, stim_channel=stim_ch)

        inst_data = inst.copy().get_data()
        if mode == 'spline':
            ch_ind = pick_types(inst.info, meg=self.meg, eeg=self.eeg,
                                eog=True, ecg=True, stim=False)
        else:
            ch_ind = pick_types(inst.info, meg=self.meg, eeg=self.eeg,
                                eog=True, ecg=True, stim=False)
        if isinstance(inst, BaseEpochs):
            event_ind = np.where(inst.times==0)[0][0]
            interp_data = np.zeros(inst_data.shape)
            for i in range(inst_data.shape[0]):
                epoch_data = inst_data[i]
                interp_data[i] = self._interpolate(epoch_data, ch_ind,
                                                   [event_ind], npoint_art,
                                                   offset, points, k)
            interp_spline = EpochsArray(interp_data, inst.info, events=inst.events,
                                        tmin=inst.tmin, verbose=False)
        else:
            interp_data = self._interpolate(inst_data, ch_ind, events[:, 0],
                                            npoint_art, offset, points, k)
            interp_spline = RawArray(interp_data, inst.info, verbose=False)
            interp_spline = interp_spline.set_annotations(inst.annotations)
        if mode == 'spline':
            interp = interp_spline
        else:
            interp._data[ch_ind] = interp_spline._data[ch_ind]
        return interp, inst, events


    def plotInterpolateArtifact(self, event, use_raw=True, keyword=None, mode='linear',
                                npoint_art=1, offset=0, points=10, k=3, show=True,
                                ylim=[-5e-4, 5e-4], tmin=-0.03, tmax=0.03,
                                baseline=(-1.1,-0.1)):
        from mne import Epochs, pick_types
        if mode not in ('spline', 'linear', 'window'):
            raise ValueError('Mode must be spline/linear/window')
        interp2, inst, events = \
            self.interpolateArtifact(event, use_raw=use_raw, keyword=keyword,
                                     mode=mode, npoint_art=npoint_art,
                                     offset=offset, points=points, k=k)
        if use_raw:
            # Note no baseline is used but epochs are only used for visualization
            epochs = Epochs(inst, events, preload=True, verbose=False, detrend=1,
                            tmin=tmin, tmax=tmax, baseline=None)
            interp = Epochs(interp2, events, preload=True, verbose=False, detrend=1,
                            tmin=tmin, tmax=tmax, baseline=None)
        else:
            epochs = inst.copy()
            interp = interp2.copy()
            epochs = epochs.crop(tmin=tmin, tmax=tmax)
            interp = interp.crop(tmin=tmin, tmax=tmax)
        times = epochs.times
        event_ind = np.where(times==0)[0][0]
        left_edge = event_ind-offset
        right_edge = event_ind+npoint_art-offset
        base0 = range(left_edge-points, left_edge)
        base1 = range(right_edge, right_edge+points)
        interp_ind = range(left_edge, right_edge)
        other0 = range(0, left_edge-points)
        other1 = range(right_edge+points, len(times))
        this_ind = pick_types(epochs.info, meg=self.meg, eeg=self.eeg)
        aux_ind = pick_types(epochs.info, meg=False, ecg=True, eog=True)
        evoked_data = epochs.get_data().mean(axis=0)
        interp_data = interp.get_data().mean(axis=0)
        fig,(ax1, ax2) = plt.subplots(2, 1)
        fig.suptitle('Evoked Colored By Interpolation Parameters')
        for ax, ch_ind in zip([ax1, ax2],[this_ind, aux_ind]):
            for ch, interp_ch in zip(evoked_data[ch_ind], interp_data[ch_ind]):
                ch_mean = ch[list(other0)+list(other1)].mean(axis=0)
                interp_ch_mean = interp_ch[list(other0)+list(other1)].mean(axis=0)
                ax.plot(times[other0], ch[other0]-ch_mean, color='k')
                ax.plot(times[other1], ch[other1]-ch_mean, color='k')
                ax.plot(times[interp_ind], ch[interp_ind]-ch_mean, color='r')
                ax.plot(times[interp_ind], interp_ch[interp_ind]-
                                          interp_ch_mean, color='g')
                ax.plot(times[base0], ch[base0]-ch_mean, color='b')
                ax.plot(times[base1], ch[base1]-ch_mean, color='b')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('V')
            ax.set_ylim(ylim)
        fig.savefig(self._fname('plots', 'interp', 'jpg', event, keyword,
                                'raw'*use_raw, 'points_%i' % points,
                                'offset_%i' % offset, 'npoint_art_%i' % npoint_art))
        self._show_fig(fig, show)
        return interp2, inst


    def _interpolate(self, this_data, ch_ind, events, npoint_art, offset, points, k):
        from scipy import interpolate
        for event in events:
            left_edge = event-offset
            right_edge = event+npoint_art-offset
            x = np.concatenate((range(left_edge-points, left_edge),
                                range(right_edge, right_edge+points)))
            xnew = np.arange(left_edge, right_edge)
            y = np.concatenate((this_data[ch_ind, left_edge-points:left_edge],
                                this_data[ch_ind, right_edge:right_edge+points]),
                                axis=1)
            ynew = np.zeros((y.shape[0], xnew.shape[0]))
            for i in range(len(y)):
                tck = interpolate.splrep(x, y[i,:], s=0.0, k=k)
                ynew[i,:] = interpolate.splev(xnew, tck, der=0)
            this_data[ch_ind, left_edge:right_edge] = ynew
        return this_data


    def applyInterpolation(self, inst, event=None, keyword_out=None, overwrite=False):
        from mne import BaseEpochs
        from mne.io import BaseRaw
        if isinstance(inst, BaseEpochs):
            if event is None:
                raise ValueError('The event of the epochs must be provided.')
            if self._has_epochs(event, keyword=keyword_out) and not overwrite:
                self._overwrite_error('Epochs', event=event, keyword=keyword_out)
            self._save_epochs(inst, event, keyword=keyword_out)
        elif isinstance(inst, BaseRaw):
            if self._has_raw(keyword=keyword_out) and not overwrite:
                self._overwrite_error('Raw', keyword=keyword_out)
            self._save_raw(inst, keyword=keyword_out)


    def filterEpochs(self, event, keyword_in=None, keyword_out=None,
                     h_freq=None, l_freq=None, overwrite=False):
        keyword_out = keyword_out if keyword_out is not None else keyword_in
        if self._has_epochs(event, keyword=keyword_out) and not overwrite:
            self._overwrite_error('Epochs', event=event, keyword=keyword_out)
        epochs = self._load_epochs(event, keyword=keyword_in)
        epochs = epochs.copy().filter(h_freq=h_freq, l_freq=l_freq)
        self._save_epochs(epochs, event, keyword=keyword_out)


    def filterRaw(self, keyword_in=None, keyword_out=None, l_freq=None, h_freq=None,
                  maxwell=False, overwrite=False):
        keyword_out = keyword_out if keyword_out is not None else keyword_in
        fname = self._fname('raw', 'raw', 'fif', keyword_out)
        if self._has_raw(keyword=keyword_out) and not overwrite:
            self._overwrite_error('Raw', keyword=keyword)
        raw = self._load_raw(keyword=keyword_in)
        if maxwell:
            from mne.preprocessing import maxwell_filter
            raw = maxwell_filter(raw)
        else:
            raw = raw.filter(l_freq=l_freq, h_freq=h_freq)
        if keyword_out:
            ica = False
        self._save_raw(raw, keyword=keyword_out)


    def sourceBootstrap(self, event, condition=None, values=None,
                        keyword_in=None, keyword_out=None,
                        snr=1.0, method='dSPM', pick_ori='normal', tfr=True,
                        itc=False, fmin=4, fmax=150, nmin=2, nmax=75, steps=32,
                        bands={'theta':(4, 8), 'alpha':(8, 15),
                               'beta':(15, 35), 'low-gamma':(35, 80),
                               'high-gamma':(80, 150)},
                        Nboot=1000, nave=50, seed=13, n_jobs=10,
                        use_fft=True, mode='same', batch=10, overwrite=False):
        ''' You need enough bootstraps to get a good normal distribution
            of your condition value means, 250 seems to do good. Nave is
            a tradeoff between more extreme values and lower snr of source
            estimates. Nboot is better the greater the number but 100 is
            approximately 40 GB with tfr'''
        from mne.time_frequency.tfr import cwt
        from mne.minimum_norm import apply_inverse
        from mne.time_frequency import morlet
        epochs = self._load_epochs(event, keyword=keyword_in)
        bl_epochs = self._load_epochs('Baseline', keyword=keyword_in)
        keyword_out = keyword_in if keyword_out is None else keyword_out
        fname = self._fname('analyses', 'bootstrap', 'npz', event, keyword_out)
        if os.path.isfile(fname) and not overwrite:
            self._overwrite_error('Bootstraps', event=event, keyword=keyword_out)
        if condition is None:
            print('Making bootstrapped source estimates using evokeds ' +
                  'of size %i to prepare to correlate the average ' % Nave +
                  'value of each bootstrap for a given condition with ' +
                  'the source estimate...')
        else:
            values = self._default_values(condition, values=values)
            value_indices = self._get_indices(epochs, condition, values)
            Nave = min([len(value_indices[value]) for value in value_indices])
            epochs = epochs.drop([i for i in range(len(epochs)) if not 
                                  any([i in value_indices[value] 
                                       for value in values])])
            print('Making bootstrapped source estimates using evokeds ' +
                  'of size %s, the downsampled number of trials for ' % Nave +
                  '%s with values %s ' % (condition, ', '.join(values)) +
                  'to prepare to compare the observed source estimate ' + 
                  'for each value to the bootstrap distribution...')
        freqs = np.logspace(np.log10(fmin), np.log10(fmax), steps)
        n_cycles = np.logspace(np.log10(nmin), np.log10(nmax), steps)
        band_inds = {band:[i for i, f in enumerate(freqs) if
                     f >= bands[band][0] and f <= bands[band][1]]
                     for band in bands if [i for i, f in enumerate(freqs) if
                     f >= bands[band][0] and f <= bands[band][1]]}
        np.random.seed(seed)
        removal_indices = []
        for j, i in enumerate(epochs.events[:, 2]):
            if not i in bl_epochs.events[:, 2]:
                removal_indices.append(j)
        epochs = epochs.drop(removal_indices)
        bootstrap_indices = np.random.randint(0, len(epochs),(Nboot, nave))
        bem, source, coord_trans, lambda2, epochs, bl_epochs, fwd = \
            self._source_setup(event, snr, epochs, bl_epochs)
        events = epochs.events[:, 2]
        bl_events = bl_epochs.events[:, 2]

        stcs = np.memmap(op.join(self.subjects_dir,
                                 'sb_%s_%s_workfile' % (event, keyword_out)),
                         dtype='float64', mode='w+',
                         shape=(batch, fwd['nsource'], len(epochs.times)))
        if tfr:
            Ws = morlet(epochs.info['sfreq'], freqs, n_cycles=n_cycles,
                        zero_mean=False)
            powers = {band:np.memmap(op.join(self.subjects_dir,
                                             'sb_tfr_%s_%s_workfile' % (event, keyword_out)),
                                     dtype='float64', mode='w+',
                                     shape=(batch, fwd['nsource'], len(epochs.times)))
                      for band in bands}
            if itc:
                itcs = {band:np.memmap(op.join(self.subjects_dir,
                                               'sb_tfr_%s_%s_workfile' % (event, keyword_out)),
                                        dtype='float64', mode='w+',
                                        shape=(batch, fwd['nsource'], len(epochs.times)))
                      for band in bands}
            else:
                itcs = None
        else:
            powers = itcs = Ws = None
        mins = range(0, nboot-batch +1, batch)
        maxs = range(batch, nboot+1, batch)
        for i_min, i_max in zip(mins, maxs):
            print('Computing bootstraps %i to %i' % (i_min, i_max))
            fname2 = self._fname('analyses', 'bootstrap', 'npz',
                                '%i-%i' % (i_min, i_max), event, keyword_out)
            if os.path.isfile(fname2) and not overwrite:
                continue
            for i, k in enumerate(tqdm(range(i_min, i_max))):
                indices = bootstrap_indices[k]
                bl_indices = [np.where(bl_events == j)[0][0]
                              for j in events[indices]]
                inv = self._generate_inverse(epochs[indices], fwd,
                                             bl_epochs[bl_indices],
                                             lambda2, method, pick_ori)
                evoked = epochs[indices].average()
                stc = apply_inverse(evoked, inv, lambda2=lambda2, method=method,
                                    pick_ori=pick_ori)
                stcs[i] = stc.data[:]
                if tfr:
                    this_tfr = cwt(stc.data.copy(), Ws, use_fft=use_fft,
                                   mode=mode)
                    power = (this_tfr * this_tfr.conj()).real
                    if itc:
                        this_itc = np.angle(this_tfr) # Inter-trial Coherence
                    for band, inds in band_inds.items():
                        powers[band][i] = power[:, inds].mean(axis=1)
                        if itc:
                            itcs[band][i] = this_itc[:, inds].mean(axis=1)
            np.savez_compressed(fname2, stcs=stcs)
            if tfr:
                for band in bands:
                    fname3 = self._fname('analyses',
                                         'bootstrap_power_%s' % band,
                                         'npz', '%i-%i' % (i_min, i_max), event,
                                         keyword_out)
                    np.savez_compressed(fname3, powers=powers[band])
                    if itc:
                        fname4 = self._fname('analyses',
                                             'bootstrap_itc_%s' % band,
                                             'npz', '%i-%i' % (i_min, i_max), event,
                                             keyword_out)
                        np.savez_compressed(fname4, itcs=itcs[band])
        if condition is not None:
            value_indices = self._get_indices(epochs, condition, values)
            for value in values:
                print('Making base source estimate for %s' % value)
                fname2 = self._fname('analyses', 'bootstrap', 'npz',
                                     event, condition, value, keyword_out)
                indices = value_indices[value]
                bl_indices = [np.where(bl_events == j)[0][0]
                              for j in events[indices]]
                inv = self._generate_inverse(epochs[indices], fwd,
                                             bl_epochs[bl_indices],
                                             lambda2, method, pick_ori)
                evoked = epochs[indices].average()
                stc = apply_inverse(evoked, inv, lambda2=lambda2, method=method,
                                    pick_ori=pick_ori)
                np.savez_compressed(fname2, stc=stc.data[:])
                if tfr:
                    this_tfr = cwt(stc.data.copy(), Ws, use_fft=use_fft,
                                   mode=mode)
                    power = (this_tfr * this_tfr.conj()).real
                    if itc:
                        this_itc = np.angle(this_tfr) # Inter-trial Coherence
                    for band, inds in band_inds.items():
                        fname3 = self._fname('analyses',
                                             'bootstrap_power_%s' % band,
                                             'npz', event, condition, value,
                                             keyword_out)
                        np.savez_compressed(fname3, power=power[:, inds].mean(axis=1))
                        if itc:
                            fname4 = self._fname('analyses',
                                             'bootstrap_itc_%s' % band,
                                             'npz', event, condition, value,
                                             keyword_out)
                            np.savez_compressed(fname4, itc=this_itc[:, inds].mean(axis=1))
        inv = self._generate_inverse(epochs, fwd, bl_epochs, lambda2, method,
                                     pick_ori)
        evoked = epochs.average()
        stc = apply_inverse(evoked, inv, lambda2=lambda2, method=method,
                            pick_ori=pick_ori)
        np.savez_compressed(fname, events=events, batch=batch, tfr=tfr, itc=itc,
                            freqs=freqs, n_cycles=n_cycles, bands=bands,
                            bootstrap_indices=bootstrap_indices)
        self._save_source(stc, event, 'Bootstrap', 'base', keyword=keyword_out)


    def sourcePermutation(self, event, condition, values=None, keyword_in=None,
                          keyword_out=None, baseline=(-0.5,-0.1),
                          n_permutations=2000, correlation=False,
                          overwrite=False):
        ''' baseline only supported in source window (this also means
            locked to the same event as the source estimate) due to
            normalization issues.
            n_permutations = 1000 takes ~5 hours with tfr'''
        from scipy.stats import linregress
        keyword_out = keyword_in if keyword_out is None else keyword_out
        fname_in = self._fname('analyses', 'bootstrap', 'npz', event,
                               keyword_in)
        analysis_name = 'correlation' if correlation else 'permutation'
        fname_out = self._fname('analyses', analysis_name, 'npz', event,
                                keyword_out)
        if not os.path.isfile(fname_in):
            raise ValueError('Bootstraps must be computed first' +
                             '(check that keywords match)')
        values = self._default_values(condition, values=values)
        if correlation:
            if self._has_source(event, condition, 'permutation',
                                keyword=keyword_out) and not overwrite:
                self._overwrite_error('Correlation', event=event,
                                      condition=condition, keyword=keyword_out)
        else:
            if all([self._has_source(event, condition, value+'_correlation',
                                     keyword=keyword_out) for value in values]):
                self._overwrite_error('Permutation', event=event, condition=condition,
                                      values=values, keyword=keyword_out)
        f = np.load(fname_in)
        bootstrap_indices = f['bootstrap_indices']
        Nboot, nave = bootstrap_indices.shape
        batch = f['batch'].item()
        bands = f['bands'].item()
        tfr = f['tfr'].item()
        itc = f['itc'].item()
        events = f['events']
        df = read_csv(self.behavior)
        stc = self._load_source(event, 'Bootstrap', 'base', keyword=keyword_in)
        if baseline[0] < stc.tmin or baseline[1] > stc.times[-1]:
            raise ValueError('Baseline outside time range')
        nSRC, nTIMES = stc.shape
        bl_indices = np.where((stc.times >= baseline[0]) &
                              (stc.times < baseline[1]))[0]
        permutation_indices = np.random.randint(0, nboot,(n_permutations, nboot))
        stc_copy = stc.copy()
        stc_copy.data.fill(0)
        #
        stcs = np.memmap('sb_%s_%s_workfile' % (event, keyword_out),
                         dtype='float64', mode='w+',
                         shape=(Nboot, nSRC, nTIMES))
        stc_result = stc_copy.copy()
        bl_dist = np.zeros((stc.data.shape[0], n_permutations))
        if tfr:
            powers = {band:np.memmap(('sb_power_%s_%s_%s_workfile'
                                      % (event, keyword_out, band)),
                                     dtype='float64', mode='w+',
                                     shape=(Nboot, nSRC, nTIMES))
                      for band in bands}
            power_result = {band:stc_copy.copy() for band in bands}
            power_bl_dist = {band:np.zeros((stc.data.shape[0], n_permutations))
                             for band in bands}
            if itc:
                itcs = {band:np.memmap(('sb_itc_%s_%s_%s_workfile'
                                        % (event, keyword_out, band)),
                                        dtype='float64', mode='w+',
                                        shape=(Nboot, nSRC, nTIMES))
                        for band in bands}
                itc_result = {band:stc_copy.copy() for band in bands}
                itc_bl_dist = {band:np.zeros((stc.data.shape[0], n_permutations))
                               for band in bands}
        mins = range(0, nboot-batch +1, batch)
        maxs = range(batch, nboot+1, batch)
        for i_min, i_max in zip(mins, maxs):
            print('Combining bootstraps %i to %i source' % (i_min, i_max), end='')
            fname2 = self._fname('analyses', 'bootstrap', 'npz',
                                 '%i-%i' % (i_min, i_max), event, keyword_in)
            stcs2 = np.load(fname2)['stcs']
            for i, j in enumerate(range(i_min, i_max)):
                stcs[j] = stcs2[i]
            del stcs2
            if tfr:
                for band in bands:
                    print(' %s tfr' % band, end='')
                    fname3 = self._fname('analyses',
                                         'bootstrap_power_%s' % band,
                                         'npz', '%i-%i' % (i_min, i_max), event,
                                         keyword_in)
                    powers2 = np.load(fname3)['powers']
                    if itc:
                        print(' %s itc' % band, end='')
                        fname4 = self._fname('analyses', 'bootstrap_itc_%s' % band,
                                             'npz', '%i-%i' % (i_min, i_max), event,
                                             keyword_in)
                        itcs2 = np.load(fname4)['itcs']
                    for i, j in enumerate(range(i_min, i_max)):
                        powers[band][j] = powers2[i]
                        if itc:
                            itcs[band][j] = itcs2[i]
                    del powers2
                    if itc:
                        del itcs2
            print(' Done.')

        if correlation:
            bootstrap_conditions = []
            for i in range(Nboot):
                bootstrap_conditions.append(
                    np.nanmean(np.array([df[condition][j]
                                        for j in bootstrap_indices[i]])))
            # Get baseline distribution
            for s_ind in tqdm(range(nSRC)):
                s_data = np.array(stcs[:, s_ind])
                if tfr:
                    power_s_data = {band:powers[band][:, s_ind] for band in bands}
                    if itc:
                        itc_s_data = {band:itcs[band][:, s_ind] for band in bands}
                for p_ind in range(n_permutations):
                    dist = s_data[:, bl_indices].mean(axis=1)
                    _,_, r,_,_ = linregress(dist[permutation_indices[p_ind]],
                                           bootstrap_conditions)
                    bl_dist[s_ind, p_ind] = r
                    if tfr:
                        for band in bands:
                            power_dist = power_s_data[band][:, bl_indices].mean(axis=1)
                            _,_, r,_,_ = linregress(power_dist[permutation_indices[p_ind]],
                                                   bootstrap_conditions)
                            power_bl_dist[band][s_ind, p_ind] = r
                            if itc:
                                itc_dist = itc_s_data[band][:, bl_indices].mean(axis=1)
                                _,_, r,_,_ = linregress(itc_dist[permutation_indices[p_ind]],
                                                       bootstrap_conditions)
                                itc_bl_dist[band][s_ind, p_ind] = r
            # Get p-values from permutation distribution
            for s_ind in tqdm(range(nSRC)):
                s_data = np.array(stcs[:, s_ind])
                if tfr:
                    power_s_data = {band:powers[band][:, s_ind] for band in bands}
                    if itc:
                        itc_s_data = {band:itcs[band][:, s_ind] for band in bands}
                for t_ind in range(nTIMES):
                    dist = s_data[:, t_ind]
                    _,_, r,_,_ = linregress(dist, bootstrap_conditions)
                    p = sum(abs(bl_dist[s_ind])>abs(r))/n_permutations
                    stc_result.data[s_ind, t_ind] = ((1.0/p)*np.sign(r) if p > 0 else
                                                    (1.0/n_permutations)*np.sign(r))
                    if tfr:
                        for band in bands:
                            power_dist = power_s_data[band][:, t_ind]
                            _,_, r,_,_ = linregress(power_dist, bootstrap_conditions)
                            p = sum(abs(power_bl_dist[band][s_ind])>abs(r))/n_permutations
                            power_result[band].data[s_ind, t_ind] = \
                                ((1.0/p)*np.sign(r) if p > 0 else
                                 (1.0/n_permutations)*np.sign(r))
                            if itc:
                                itc_dist = itc_s_data[band][:, t_ind]
                                _,_, r,_,_ = linregress(itc_dist, bootstrap_conditions)
                                p = sum(abs(itc_bl_dist[band][s_ind])>abs(r))/n_permutations
                                itc_result[band].data[s_ind, t_ind] = \
                                    ((1.0/p)*np.sign(r) if p > 0 else
                                     (1.0/n_permutations)*np.sign(r))
            self._save_source(stc_result, event, condition, 'correlation',
                              keyword=keyword_out)
            if tfr:
                for band in bands:
                    self._save_source(power_result[band], event, condition,
                                      'power_correlation_%s' % band,
                                      keyword=keyword_out)
                    if itc:
                        self._save_source(itc_result[band], event, condition,
                                          'itc_correlation_%s' % band,
                                          keyword=keyword_out)
        else:
            for value in values:
                fname2 = self._fname('analyses', 'bootstrap', 'npz',
                                     event, condition, value, keyword_in)
                this_stc = np.load(fname2)['stc']
                if tfr:
                    this_power = {}
                    if itc:
                        this_itc = {}
                    for band in bands:
                        fname3 = self._fname('analyses',
                                             'bootstrap_power_%s' % band,
                                             'npz', event, condition, value,
                                             keyword_in)
                        this_power[band] = np.load(fname3)['power']
                        if itc:
                            fname4 = self._fname('analyses',
                                                 'bootstrap_itc_%s' % band,
                                                 'npz', event, condition, value,
                                                 keyword_in)
                            this_itc[band] = np.load(fname3)['itc']
                # Get p-values from permutation distribution
                for s_ind in tqdm(range(nSRC)):
                    s_data = np.array(stcs[:, s_ind])
                    if tfr:
                        power_s_data = {band:powers[band][:, s_ind] for band in bands}
                        if itc:
                            itc_s_data = {band:itcs[band][:, s_ind] for band in bands}
                    for t_ind in range(nTIMES):
                        dist = s_data[:, t_ind]
                        p = sum(abs(dist)<abs(this_stc[s_ind, t_ind]))/Nboot
                        stc_result.data[s_ind, t_ind] = 1.0/p if p > 0 else Nboot
                        if tfr:
                            for band in bands:
                                power_dist = power_s_data[band][:, t_ind]
                                p = sum(abs(power_dist)<
                                        abs(this_power[band][s_ind, t_ind]))/Nboot
                                power_result[band].data[s_ind, t_ind] = \
                                    1.0/p if p > 0 else Nboot
                                if itc:
                                    itc_dist = itc_s_data[band][:, t_ind]
                                    p = sum(abs(itc_dist)<
                                            abs(this_itc[band][s_ind, t_ind]))/Nboot
                                    itc_result[band].data[s_ind, t_ind] = \
                                        1.0/p if p > 0 else Nboot
                self._save_source(stc_result, event, condition, value+'_permutation',
                                  keyword=keyword_out)
                if tfr:
                    for band in bands:
                        self._save_source(power_result[band], event, condition,
                                          value+'_power_permutation_%s' % band,
                                          keyword=keyword_out)
                        if itc:
                            self._save_source(itc_result[band], event, condition,
                                              value+'_itc_permutation_%s' % band,
                                              keyword=keyword_out)



    def noreunPhi(self, event, condition, values=None, keyword_in=None, 
                  source_keyword=None, keyword_out=None, 
                  tmin=None, tmax=None, npoint_art=0,
                  Nboot=480, alpha=0.01, downsample=True, seed=11,
                  shared_baseline=False, bl_tmin=-0.5, bl_tmax=-0.1,
                  recalculate_baseline=False, recalculate_PCI=False):
        ''' note: has to have baseline-event continuity for source space
            transformation: cannot use baseline as would be defined for
            other applications based on another event. If there is a task and the
            baseline needs to be related to another event (e.g. the start of a
            stimulus compared to the PCI being relative to the response)
            the recommended solution is longer epochs'''
        try:
            from . import pci
        except:
            print('Unable to import pci, you won\'t be able to use this analysis')
        from mne.minimum_norm import apply_inverse
        from mne import EpochsArray
        if downsample:
            np.random.seed(seed)
        keyword_out = keyword_in if keyword_out is None else keyword_out
        tmin, tmax = self._default_t(event, tmin, tmax)
        epochs = self._load_epochs(event, keyword=keyword_in)
        epochs = epochs.crop(tmin=min([tmin, bl_tmin]), tmax=max([tmax, bl_tmax]))
        if self.eeg:
            epochs = epochs.set_eeg_reference(ref_channels='average',
                                              projection=True, verbose=False)
        epochs = self._pick_types(epochs, 'eeg')
        info = epochs.info
        values = self._default_values(condition, values=values)
        value_indices = self._get_indices(epochs, condition, values)
        bl_tind = np.intersect1d(np.where(bl_tmin<=epochs.times),
                             np.where(epochs.times<=bl_tmax))
        nTR = min([len(value_indices[value]) for value in value_indices])

        def preprocess(epochs, indices, bl_tind, event, condition, value, source_keyword):
            Y = epochs[indices].get_data()
            inv, lambda2, method, pick_ori = \
                self._load_inverse(event, condition, value, keyword=source_keyword)
            nSRC = inv['nsource']
            nTIME = len(epochs.times)
            J = apply_inverse(epochs[indices].average(), inv,
                   method=method, lambda2=lambda2,
                   pick_ori=pick_ori, verbose=False).data
            basecorr = np.mean(J[:, bl_tind], axis=1)
            N0 = len(bl_tind)
            Norm = np.std(J[:, bl_tind], axis=1, ddof=1)
            J-=np.kron(np.ones((1, nTIME)), basecorr.reshape((nSRC, 1)))
            NUM = np.kron(np.ones((1, N0)), basecorr.reshape((nSRC, 1)))
            DEN = np.kron(np.ones((1, N0)), Norm.reshape((nSRC, 1)))
            return Y, J, inv, lambda2, method, pick_ori, NUM, DEN, Norm

        def downsampleIndices(indices, nTR, condition, value):
            print('Subsampling %i/%i ' % (nTR, len(indices)) +
                  'for %s %s.' % (condition, value))
            np.random.shuffle(indices)
            indices = indices[:nTR]
            indices = sorted(indices)
            return indices

        def baseline_bootstrap(Y, J, bl_tind, Norm, NUM, DEN, Nboot, alpha,
                               info, inv, lambda2, method, pick_ori):
            nTR, nCH, nTIME = Y.shape
            nSRC = J.shape[0]
            N0 = len(bl_tind)
            randontrialsT=np.random.randint(0, nTR, nTR)
            Bootstraps=np.zeros((Nboot, N0))
            for per in tqdm(range(Nboot)):
                YT=np.zeros((nTR, nCH, N0))
                for j in range(nTR):
                    randonsampT = np.random.choice(bl_tind, N0, replace=True)#np.random.randint(0, N0, N0)
                    YT[j] = Y[randontrialsT[j]][:, randonsampT]
                YTE = EpochsArray(YT, info, verbose=False)
                ET=apply_inverse(YTE.average(), inv, method=method,
                                 lambda2=lambda2, pick_ori=pick_ori,
                                 verbose=False).data
                ET=(ET-NUM)/DEN # computes a Z-value
                Bootstraps[per,:] = np.max(np.abs(ET), axis=0) # maximum statistics in space
            # computes threshold for binarization depending on alpha value
            Bootstraps=np.sort(np.reshape(Bootstraps,(Nboot*N0)))
            calpha=1-alpha
            calpha_index=int(np.floor(calpha*Nboot*N0))
            TT=Norm*Bootstraps[calpha_index]# computes threshold based on alpha set before
            Threshold=np.kron(np.ones((1, nTIME)), TT.reshape((nSRC, 1)))
            return Threshold

        def gettind(epochs, tmin, tmax, npoint_art):
            tind = np.intersect1d(np.where(tmin<=epochs.times),
                                  np.where(epochs.times<=tmax))
            return tind[npoint_art:]

        def computePCI(binJ):
            print('Computing LZ complexity...')
            ct=pci.lz_complexity_2D(binJ)
            print('Computing the normalization factor...')
            norm=pci.pci_norm_factor(binJ)
            ct=ct/norm
            print('Done.')
            return ct

        for value in ['all'] if shared_baseline else values:
            ''' Use separate baselines to threshold for significance because
            what is significant for sleep may not be for wake ect'''
            if (self._has_noreun_baseline(event, condition, value, keyword_out)
                and not recalculate_baseline):
                print('Loading pre-computed bootstraps')
            else:
                print('Running baseline bootstrap for %s' % value)
                indices = value_indices[value]
                if downsample and not shared_baseline:
                    indices = downsampleIndices(indices, nTR, condition, value)
                Y, J, inv, lambda2, method, pick_ori, NUM, DEN, Norm = \
                    preprocess(epochs, indices, bl_tind, event, condition, value,
                               source_keyword)
                Threshold = baseline_bootstrap(Y, J, bl_tind, Norm, NUM, DEN, Nboot,
                                               alpha, info, inv, lambda2, method,
                                               pick_ori)
                self._save_noreun_baseline(Threshold, epochs[indices].events[:, 2],
                                           bl_tmin, bl_tmax, Nboot, alpha, event,
                                           condition, value, keyword_out)
        # PCI by value
        for value in values:
            if (self._has_noreun_PCI(event, condition, value, keyword_out)
                 and not recalculate_PCI):
                print('PCI already computed')
            else:
                indices = value_indices[value]
                if downsample:
                    indices = downsampleIndices(indices, nTR, condition, value)
                Threshold, events, bl_tmin, bl_tmax, Nboot, alpha = \
                    self._load_noreun_baseline(event, condition,
                                               'all' if shared_baseline else value,
                                               keyword_out)
                Y, J, inv, lambda2, method, pick_ori, NUM, DEN, Norm = \
                    preprocess(epochs, indices, bl_tind, event, condition, value,
                               source_keyword)
                tind = gettind(epochs, tmin, tmax, npoint_art)
                # determines sources matrices
                binJ=np.array(np.abs(J)>Threshold, dtype=int)
                # rank the activity matrix - use mergesort that yields same results of Matlab
                Irank=np.argsort(np.sum(binJ, axis=1), kind='mergesort')
                binJrank=np.copy(binJ)
                binJrank=binJ[Irank,:]
                binJ=binJrank[:, tind]
                print('Running Lempel-Ziv compression for %s' % value)
                ct = computePCI(binJ)
                self._save_noreun_PCI(ct, binJ, Y, J, tmin, tmax, npoint_art,
                                      event, condition, value, keyword_out)


    def _has_noreun_baseline(self, event, condition, value, keyword=None):
        fname = self._fname('analyses', 'Threshold', 'npz', keyword,
                            event, condition, value)
        return op.isfile(fname)


    def _load_noreun_baseline(self, event, condition, value, keyword=None):
        fname = self._fname('analyses', 'Threshold', 'npz', keyword,
                            event, condition, value)
        if op.isfile(fname):
            self._file_loaded('Bootstrap Threshold', event=event,
                              condition=condition, value=value,
                              keyword=keyword)
            f = np.load(fname)
            return (f['Threshold'], f['events'], f['bl_tmin'].item(),
                        f['bl_tmax'].item(), f['Nboot'].item(), f['alpha'].item())
        else:
            self._no_file_error('Bootstrap Threshold', event=event,
                                condition=condition, value=value,
                                keyword=keyword)


    def _save_noreun_baseline(self, Threshold, events, bl_tmin, bl_tmax, Nboot, alpha,
                              event, condition, value, keyword=None):
        self._file_saved('Bootstrap Threshold', event=event,
                         condition=condition, value=value,
                         keyword=keyword)
        fname = self._fname('analyses', 'Threshold', 'npz', keyword,
                            event, condition, value)
        np.savez_compressed(fname, Threshold=Threshold, bl_tmin=bl_tmin,
                            bl_tmax=bl_tmax, events=events, Nboot=Nboot,
                            alpha=alpha)


    def _has_noreun_PCI(self, event, condition, value, keyword=None):
        fname = self._fname('analyses', 'PCI', 'npz', keyword,
                            event, condition, value)
        return op.isfile(fname)


    def _load_noreun_PCI(self, event, condition, value, keyword=None):
        fname = self._fname('analyses', 'PCI', 'npz', keyword,
                            event, condition, value)
        if op.isfile(fname):
            self._file_loaded('PCI', event=event, condition=condition,
                                value=value, keyword=keyword)
            f = np.load(fname)
        else:
            self._no_file_error('PCI', event=event, condition=condition,
                                value=value, keyword=keyword)
        return (f['ct'], f['binJ'], f['Y'], f['J'],
                f['tmin'].item(), f['tmax'].item(), f['npoint_art'].item())


    def _save_noreun_PCI(self, ct, binJ, Y, J, tmin, tmax, npoint_art,
                         event, condition, value, keyword=None):
        self._file_saved('PCI', event=event, condition=condition,
                         value=value, keyword=keyword)
        fname = self._fname('analyses', 'PCI', 'npz', keyword,
                            event, condition, value)
        np.savez_compressed(fname, ct=ct, binJ=binJ, Y=Y, J=J, tmin=tmin, tmax=tmax,
                            npoint_art=npoint_art)


    def plotNoreunPCI(self, event, condition, values=None, keyword=None,
                      ssm=True, pci=True, evoked='Sensor', evoked_res=0.01,
                      evoked_tmin=None, evoked_tmax=None,
                      downsampled=True, shared_baseline=False,
                      fontsize=12, wspace=0.4, linewidth=4, show=True):
        values = self._default_values(condition, values=values)
        if len(values) > 1:
            fig, axs = plt.subplots(2, len(values))
        else:
            fig, axs = plt.subplots(2, 1)
            axs = axs[:, np.newaxis]
        fig.set_size_inches(8*len(values), 8)
        fig.subplots_adjust(wspace=wspace, left=0.2, right=0.8)
        yMAX = 0
        for i, value in enumerate(values):
            ct, binJ, Y, J, tmin, tmax, npoint_art = \
                self._load_noreun_PCI(event, condition, value, keyword)
            Threshold, events, bl_tmin, bl_tmax, nboot, alpha = \
                    self._load_noreun_baseline(event, condition,
                                               'all' if shared_baseline else value,
                                               keyword)
            start = float(npoint_art)/(npoint_art + ct.shape[0])*(tmax-tmin)
            if ct.max() > yMAX:
                yMAX = ct.max()
            if ssm:
                ax = axs[0, i].twinx() if pci else axs[i]
                ax.zorder = 0
                nSRC, nTIME = binJ.shape
                ax.imshow(binJ, extent=[0, nTIME, 0, nSRC],
                          aspect='auto', cmap='Greys')
                ax.set_ylim(bottom=-10, top=nSRC+10)
                ax.set_ylabel('Sources Ranked by Activity', fontsize=fontsize)
            if pci:
                ax = axs[0, i]
                ax.zorder = 1
                ax.patch.set_alpha(0)
                ax.plot(range(ct.shape[0]), ct, 'y-',
                        linewidth=linewidth, alpha=1, zorder=1)
                ax.plot(range(ct.shape[0]), ct, 'b-',
                        linewidth=linewidth/2, alpha=1, zorder=2)
                ax.set_ylabel('PCI', fontsize=fontsize)
            title = ('%s' % value + (', PCI=%2.2g' % ct[-1])*pci +
                     ', %i Trials' % Y.shape[0])
            ax.set_title(title, fontsize=fontsize)
            ax.set_xlabel('Time (s)', fontsize=fontsize)
            ax.set_xlim([-1, ct.shape[0] + 1])
            ax.set_xticks(np.linspace(0, ct.shape[0], 5))
            ax.set_xticklabels(np.round(np.linspace(start, tmax, 5), 2))
            if evoked:
                ax = axs[1, i]
                if evoked.title() == 'Source':
                    ax.plot(np.linspace(min([tmin, bl_tmin]), max([tmax, bl_tmax]), J.shape[1]),
                            J[::int(1/evoked_res)].T)
                    ax.set_ylim(top=max([0, np.quantile(J, 0.99)*2]),
                            bottom=min([0, np.quantile(J, 0.01)*2])) #for TMS pulse not to swamp
                    ax.set_ylabel('Source Estimate Activity', fontsize=fontsize)
                elif evoked.title() == 'Sensor':
                    ax.plot(np.linspace(min([tmin, bl_tmin]), max([tmax, bl_tmax]), Y.shape[2]),
                            1e6*Y.mean(axis=0).T)
                    ax.set_ylim(top=max([0, 1e6*Y.mean(axis=0).max()*1.1]),
                                bottom=min([0, 1e6*Y.mean(axis=0).min()*1.1])) #for TMS pulse not to swamp
                    ax.set_ylabel(r'$\mu$V', fontsize=fontsize)
                ax.set_xlabel('Time (s)', fontsize=fontsize)
                evoked_tmin = min([tmin, bl_tmin]) if evoked_tmin is None else evoked_tmin
                evoked_tmax = max([tmax, bl_tmax]) if evoked_tmax is None else evoked_tmax
                ax.set_xticks(np.linspace(evoked_tmin, evoked_tmax, 5))
                ax.set_xticklabels(np.round(np.linspace(evoked_tmin, evoked_tmax, 5), 2))
                ax.set_xlim([evoked_tmin, evoked_tmax])

        if pci: [ax.set_ylim(top=yMAX*1.05) for ax in axs[0]]
        title = ('%s %s' % (event, condition) + ' Significant Sources'*ssm +
                 ' and'*(pci and ssm) + ' PCI'*pci +
                 ' %s' % keyword * (keyword is not None))
        fig.suptitle(title, fontsize=fontsize)
        fig.savefig(self._fname('plots', 'noreun_phi_plot', 'jpg',
                                keyword, event, condition,*values))
        self._show_fig(fig, show)


    def raw2mat(self, keyword=None, ch=None):
        from scipy.io import savemat
        raw = self._load_raw(keyword=keyword)
        if ch is None:
            ch_dict = self._get_ch_dict(raw)
        else:
            ch_dict = {raw.info['ch_names'].index(ch): ch}
        raw_data = raw.get_data()
        data_dict = {}
        for ch in ch_dict:
            data_dict[ch_dict[ch]] = raw_data[ch]
        savemat(self._fname('raw', 'data', 'mat', keyword), data_dict)


    def _has_wc(self, event, condition, value, band,
                data_type=None, keyword=None):
        fname = self._fname('analyses', 'WaveletConnectivity',
                            'npz', keyword, event, condition,
                            value, band, data_type)
        return op.isfile(fname)


    def _load_wc(self, event, condition, value, band,
                 data_type=None, keyword=None):
        fname = self._fname('analyses', 'WaveletConnectivity',
                            'npz', keyword, event, condition, value,
                            band, data_type)
        if os.path.isfile(fname):
            self._file_loaded('Wavelet Connectivity', band,
                              event=event, condition=condition,
                              value=value, data_type=data_type,
                              keyword=keyword)
            f = np.load(fname)
        else:
            self._no_file_error('Wavelet Connectivity', band,
                                event=event, condition=condition,
                                value=value, data_type=data_type,
                                keyword=keyword)
        return f['con']


    def _save_wc(self, con, event, condition, value, band,
                 data_type=None, keyword=None):
        self._file_saved('Wavelet Connectivity', band, event=event,
                         condition=condition, value=value,
                         data_type=data_type, keyword=keyword)
        fname = self._fname('analyses', 'WaveletConnectivity', 'npz',
                            keyword, event, condition, value, band, data_type)
        np.savez_compressed(fname, con=con)


    def waveletConnectivity(self, event, condition, values=None,
                            keyword_in=None, keyword_out=None, downsample=True,
                            seed=13, threshold=0.05, min_dist=0.05,
                            tmin=None, tmax=None, vis_tmin=None, vis_tmax=None,
                            fmin=4, fmax=150, nmin=2, nmax=75, steps=32, gif_combine=True,
                            method='pli', tube_radius=0.001, time_scalar=10,
                            n_jobs=5, stc=True, fps=20,
                            bands={'theta': (4, 8), 'alpha': (8, 15),
                                   'beta': (15, 35), 'low-gamma': (35, 80),
                                   'high-gamma': (80, 150)},
                            views=['llat', 'cau', 'dor', 'ven', 'fro', 'lfpar',
                                   'rlat', 'rcpar'],
                            cmap='Reds', overwrite=False):
        try:
            from mayavi import mlab
        except:
            print('Unable to import mayavi')
        try:
            import moviepy.editor as mpy
        except: 
            print('Unable to import MoviePy, connectivity animations will not be available')
        from .gif_combine import combine_gifs
        from scipy import linalg
        from mne.connectivity import spectral_connectivity
        if downsample:
            np.random.seed(seed)
        keyword_out = keyword_in if keyword_out is None else keyword_out
        frequencies = np.logspace(np.log10(fmin), np.log10(fmax), steps)
        n_cycles = np.logspace(np.log10(nmin), np.log10(nmax), steps)
        tmin, tmax = self._default_t(event, tmin, tmax, buffered=True)
        vis_tmin, vis_tmax = self._default_t(event, vis_tmin, vis_tmax) #no buffer
        epochs = self._load_epochs(event, keyword=keyword_in)
        epochs = epochs.crop(tmin=tmin, tmax=tmax)
        values = self._default_values(condition, values=values)
        value_indices = self._get_indices(epochs, condition, values)
        nTR = min([len(value_indices[value]) for value in value_indices])
        sfreq = epochs.info['sfreq']
        if self.eeg:
            epochs = epochs.set_eeg_reference(ref_channels='average',
                                              projection=True, verbose=False)
        if bands == None:
            bands = {('%.1f' % f):(f, f) for f in frequencies}
        for dt in self._get_data_types():
            print(dt)
            epo = self._pick_types(epochs, dt)
            sens_loc = np.array([ch['loc'][:3] for ch in epo.info['chs']])
            #time_factor = int(len(epo.times)/(time_dilation*(tmax-tmin)*fps))
            duration = (vis_tmax-vis_tmin)*sfreq/(fps*time_scalar)
            vis_tmin_ind = int((vis_tmin-tmin)*sfreq)
            if time_scalar < 1:
                raise ValueError('The minimum time scale is 1, which would ' +
                                 'display all connectivity points')
            elif time_scalar > len(epo.times):
                raise ValueError('Too many frames per second and/or time scaling, ' + 
                                 'not enough connectivity time points')
            for band_name, (fmin, fmax) in bands.items():
                print(band_name)
                if not any([f for f in frequencies if f >= fmin and f <= fmax]):
                    print('No frequencies for this band')
                    continue
                freqs = np.array([f for f in frequencies 
                                  if f >= fmin and f <= fmax])
                n_cs = np.array([n for f, n in zip(frequencies, n_cycles) if
                                 f >= fmin and f <= fmax])
                for value in values:
                    print(value)
                    indices = value_indices[value]
                    if not indices:
                        print('No epochs for this value')
                        continue
                    if downsample:
                        print('Subsampling %i/%i %s %s.' % (nTR, len(indices),
                                                           condition, value))
                        np.random.shuffle(indices)
                        indices = indices[:nTR]
                    if (self._has_wc(event, condition, value, band_name,
                                     dt, keyword_out) and 
                        not overwrite):
                        con = self._load_wc(event, condition, value, band_name,
                                            data_type=dt, keyword=keyword_out)
                    else:
                        con, av_freqs, times, n_epochs, n_tapers = \
                            spectral_connectivity(epo[indices], method=method,
                                                  mode='cwt_morlet', sfreq=sfreq,
                                                  fmin=fmin, fmax=fmax, faverage=True,
                                                  tmin=tmin, cwt_freqs=freqs,
                                                  cwt_n_cycles=n_cs,
                                                  n_jobs=n_jobs)
                        con = con.squeeze()
                        self._save_wc(con, event, condition, value, band_name,
                                      data_type=dt, keyword=keyword_out)
                    if not overwrite and ((gif_combine and 
                        op.isfile(self._fname('plots', 'connectivity', 'gif',
                                              event, condition, value,
                                              band_name, dt, keyword_out,
                                              *views))) or
                        all([op.isfile(self._fname('plots', 'connectivity', 'gif',
                                                   event, condition, value,
                                                   band_name, dt, keyword_out,
                                                   view)) for view in views])):
                        continue
                    con_sorted = np.sort(con, axis=None)
                    vmin = con_sorted[-int(threshold/len(epo.times)*con.size)]
                    vmax = con_sorted[-1]
                    indices = np.where(con >= vmin)
                    nodes = {t:{} for t in range(len(epo.times))}
                    for i, j, t in zip(*indices):
                        if linalg.norm(sens_loc[i]-sens_loc[j]) > min_dist:
                            nodes[t][(i, j)] = con[i, j, t]
                    gif_names = []
                    for view in views + ['cbar']:
                        print(view)
                        fig = mlab.figure(size=(600, 600), bgcolor=(0, 0, 0))
                        fig.scene._lift() #weird bug https://github.com/enthought/mayavi/issues/702
                        mlab_view = view_to_mlab(view)
                        scale_points = mlab.plot3d([0, 1],[0, 1],[0, 1],[vmin, vmin],
                                               vmin=vmin, vmax=vmax,
                                               tube_radius=tube_radius,
                                               colormap=cmap)
                        def make_frame(t):
                            mlab.clf()
                            t_ind = vis_tmin_ind + int(t*fps)*time_scalar
                            print(t_ind)
                            if view == 'cbar' or not gif_combine:
                                mlab.scalarbar(scale_points, title=('Phase Lag Index (PLI), ' +
                                               'time=%.2f' % epo.times[t_ind]),
                                               nb_labels=4)
                            if view != 'cbar':
                                pts = mlab.points3d(sens_loc[:, 0],
                                                    sens_loc[:, 1], 
                                                    sens_loc[:, 2],
                                                    color=(1, 1, 1),
                                                    opacity=1,
                                                    scale_factor=0.005)
                                for node, val in tqdm(nodes[t_ind].items()):
                                    x1, y1, z1 = sens_loc[node[0]]
                                    x2, y2, z2 = sens_loc[node[1]]
                                    points = mlab.plot3d([x1, x2], [y1, y2],
                                                         [z1, z2], [val, val],
                                                         vmin=vmin, vmax=vmax,
                                                         tube_radius=tube_radius,
                                                         colormap=cmap)
                                #points.module_manager.scalar_lut_manager.reverse_lut = True
                                mlab.view(*mlab_view)
                                mlab.move(up=-0.025)
                            return mlab.screenshot(antialiased=True)

                        anim = mpy.VideoClip(make_frame, duration=duration)
                        fname = self._fname('plots', 'connectivity', 'gif', event, condition,
                                            value, band_name, dt, keyword_out, view)
                        gif_names.append(fname)
                        anim.write_gif(fname, fps=fps)
                        mlab.close(fig)
                    if gif_combine:
                        print('Combining gifs for %s' % value)
                        anim = combine_gifs(self._fname('plots', 'connectivity', 'gif',
                                                        event, condition, value, band_name, dt,
                                                        keyword_out,*views),
                                            fps,*gif_names)


def view_to_mlab(view):
    view_dict = {'dor':(-90, 50, 0.75, np.zeros((3,))),
                 'llat':(5.6, 85, 0.75, np.zeros((3,))),
                 'rlat':(5.6,-85, 0.75, np.zeros((3,))),
                 'cau':(-85, 90, 0.75, np.zeros((3,))),
                 'ven':(-138, 177, 0.75, np.zeros((3,))),
                 'fro':(92, 71, 0.75, np.zeros((3,))),
                 'lfpar':(-45,-57, 0.75, np.zeros((3,))),
                 'rcpar':(-45, 57, 0.75, np.zeros((3,))),
                 'cbar':(92, 71, 0.75, np.zeros((3,)))}
    return view_dict[view]


def create_demi_events(raw_fname, window_size, shift, epoches_nun=0,
                       fname_out=None, overwrite=False):
    if fname_out is None and not overwrite:
        raise ValueError('No out file name specified, use \'overwrite=True\'' +
                         'to overwrite.')
    windows_length = raw.info['sfreq']*window_size
    windows_shift = raw.info['sfreq']*shift
    import math
    # T = raw._data.shape[1]
    T = raw.last_samp - raw.first_samp + 1
    if epoches_nun == 0:
        epoches_nun = int(math.floor((T - windows_length) / windows_shift + 1))
    demi_events = np.zeros((epoches_nun, 3), dtype=np.uint32)
    for win_ind in range(epoches_nun):
        demi_events[win_ind] = [win_ind * windows_shift, win_ind * windows_shift + windows_length, 0]
    # for win_ind, win in enumerate(range(0, max_time, W * 2)):
        # demi_events[win_ind * 2] = [win, win + W, 0]
        # demi_events[win_ind * 2 + 1] = [win + W + 1, win + W * 2, 1]
    # demi_events_ids = {'demi_1': 0, 'demi_2': 1}
    demi_events[:, :2] += raw.first_samp
    demi_conditions = {'demi': 0}
    return demi_events, demi_conditions


def loadMEEGbuddy(subjects_dir, subject, session=None, run=None, task=None,
                  eeg=False, meg=False, ecog=False, seeg=False):
    name = str(subject) + '_'
    name += str(session) + '_' if session is not None else ''
    name += str(run) + '_' if run is not None else ''
    name += str(task) + '_' if task is not None else ''
    for dt, v in {'meg': meg, 'eeg': eeg, 'ecog': ecog, 'seeg': seeg}.items():
        if v:
            name += dt
    file = op.join(subjects_dir, 'meta_data', name + '.json')
    if op.isfile(file):
        return MEEGbuddy(file=file)
    else:
        raise ValueError('%s file not found' % file)


def loadMEEGbuddies(subjects_dir, meg=None, eeg=None, task=None, shuffled=False,
                    seed=11):
    mbs = os.listdir(op.join(subjects_dir, 'meta_data'))
    for mb in mbs.copy():
        if (op.splitext(mb)[1] != '.json' or
            (meg is not None and (meg and not 'meg' in mb or not meg and 'meg' in mb)) or
            (eeg is not None and (eeg and not 'eeg' in mb or not eeg and 'eeg' in mb)) or
            (task is not None and not task in mb)):
            mbs.remove(mb)
    mbs = [MEEGbuddy(file=op.join(subjects_dir, 'meta_data', mb)) for mb in mbs]
    if shuffled:
        np.random.seed(seed)
        np.random.shuffle(mbs)
    return mbs


def getMEEGbuddiesBySubject(subjects_dir, meg=None, eeg=None, task=None,
                            shuffled=False, seed=11):
    data_struct = loadMEEGbuddies(subjects_dir, meg=meg, eeg=eeg, task=task,
                                  shuffled=shuffled, seed=seed)
    data_by_subject = {}
    for data in data_struct:
        if data.subject in data_by_subject:
            data_by_subject[data.subject].append(data)
        else:
            data_by_subject[data.subject] = [data]
    return data_by_subject


def source_fs_and_mne(fs_subjects_dir):
    from subprocess import call
    if 'FREESURFER_HOME' in os.environ:
        if fs_subjects_dir is None:
            raise ValueError('Must specify a directory to put the reconstructions in fs_subjects_dir')
        if not op.isdir(fs_subjects_dir):
            os.makedirs(fs_subjects_dir)
        subjects_dir = fs_subjects_dir
        os.environ['SUBJECTS_DIR'] = subjects_dir
        try:
            call(['source $FREESURFER_HOME/SetUpFreeSurfer.sh'], shell=True,
                 env=os.environ)
        except Exception as e:
            raise Exception('Freesurfer not installed or installed correctly. ' + 
                            'Make sure $FREESURFER_HOME is defined correctly.' +
                            'See surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall')
        try:
            shell = op.basename(os.environ['SHELL'])
            if shell == 'bash' or shell == 'tcsh':  # might have to change these only tested for tcsh
                call(['source $MNE_ROOT/bin/mne_setup_sh'], shell=True, env=os.environ)
            elif shell == 'csh':
                call(['source $MNE_ROOT/bin/mne_setup'], shell=True, env=os.environ)
            else:
                raise ValueError('Shell not bash or csh or not understood')
        except:
            raise Exception('MNE_C not installed or installed correctly. ' +
                            'Make sure $MNE_ROOT is defined correctly. ' +
                            'See martinos.org/mne/stable/install_mne_c.html')
    else:
        raise Exception('Feesurfer was not sourced. Please source ' +
                        'freesurfer to continue')

def recon_subject(subject, bids_dir, out_dir, fs_subjects_dir, overwrite,
                  ico, ico2, conductivity, spacing, surface):
    from subprocess import call
    os.environ['SUBJECT'] = subject
    t1f = op.join(bids_dir, 'sub-%s' % subject, 'anat', 'sub-%s_T1w.nii.gz' % subject)
    if op.isdir(op.join(out_dir, subject, 'mri')) and not overwrite:
        print('Recon already run, skipping')
    else:
        if op.isdir(op.join(out_dir, subject, 'mri')):
            os.rmdir(op.join(out_dir, subject))
        call(['recon-all -subjid %s -i %s -all' % (subject, t1f)], shell=True, env=os.environ)

    call(['mne_setup_source_space --ico %s --cps --overwrite' % ico2], shell=True, env=os.environ)
    #Organize flash if supplied
    flash = op.isfile(op.join(bids_dir, 'sub-%s' % subject, 'anat', 'sub-%s_FLASH.nii.gz' % subject))
    if flash:
        if op.isdir(op.join(fs_subjects_dir, subject, '/flash05')):
            os.rmdir(op.join(fs_subjects_dir, subject, '/flash05'))
        os.makedirs(op.join(fs_subjects_dir, subject, '/flash05'))
        os.chdir(op.join(fs_subjects_dir, subject, '/flash05'))
        flash_dir = op.join(bids_dir, 'sub-%s' % subject, 'anat')
        call(['mne_organize_dicom %s' % flash_dir], shell=True, env=os.environ)
        new_flash_dir = op.join(fs_subjects_dir, subject, 'flash05', 
                                'sub-%s_FLASH_MEFLASH_8e_05deg')  # probably not right
        call(['ln -s %s flash05' % new_flash_dir], shell=True, env=os.environ)
        call(['mne_flash_bem --noflash30'], shell=True, env=os.environ)
        call(['freeview -v %s/%s/mri/T1.mgz ' % (fs_subjects_dir, subject) + 
              '-f %s/%s/bem/flash/inner_skull.surf ' % (fs_subjects_dir, subject)+
              '-f %s/%s/bem/flash/outer_skull.surf ' % (fs_subjects_dir, subject)+
              '-f %s/%s/bem/flash/outer_skin.surf' % (fs_subjects_dir, subject)],
              shell=True, env=os.environ)
        for area in ['inner_skull', 'outer_skull', 'outer_skin']:
            link = op.join(fs_subjects_dir, subject, 'bem','%s.surf' % area)
            flash_link = op.join(fs_subjects_dir, subject, 'bem', 'flash',
                                     '%s_%s_surface' % (subject, area))
            if op.isfile(link):
                os.remove(link)
            call(['ln -s %s %s' %(flash_link, link)], shell=True, env=os.environ)
    else:
        call(['mne_watershed_bem --subject %s --atlas --overwrite' % subject], shell=True, 
             env=os.environ)
        call(['freeview -v %s/%s/mri/T1.mgz ' % (fs_subjects_dir, subject) + 
              '-f %s/%s/bem/watershed/%s_inner_skull_surface ' % (fs_subjects_dir, subject, subject) +
              '-f %s/%s/bem/watershed/%s_outer_skull_surface ' % (fs_subjects_dir, subject, subject)+ 
              '-f %s/%s/bem/watershed/%s_outer_skin_surface' % (fs_subjects_dir, subject, subject)],
              shell=True, env=os.environ)
        for area in ['inner_skull', 'outer_skull', 'outer_skin']:
            link = op.join(subjects_dir,subject, 'bem', '%s.surf' % area)
            watershed_link = op.join(fs_subjects_dir, subject, 'bem', 'watershed',
                                     '%s_%s_surface' % (subject, area))
            if op.isfile(link):
                os.remove(link)
            call(['ln -s %s %s' % (watershed_link, link)], shell=True, env=os.environ)
    # BEM
    call(['mne_setup_forward_model --subject %s --surf ' % subject + 
          '--ico %i --innershift %i' % (ico, 2 if flash else -1)], shell=True, env=os.environ)
    if not op.isfile(op.join(fs_subjects_dir, subject, 'bem',
                     '%s-5120-5120-5120-bem-sol.fif' % subject)):
        raise ValueError('Forward model failed, likely due to errors in ' + 
                         'freesurfer reconstruction and segementation. ' + 
                         'See https://github.com/ezemikulan/blender_freesurfer ' +
                         'for a workaround, functions from ' + 
                         'https://github.com/andersonwinkler/toolbox are needed too')
    os.chdir(fs_subjects_dir)

    # make head model
    call(['mkheadsurf -subjid %s' % subject], shell=True, env=os.environ)
    os.chdir(op.join(subjects_dir,subject,'bem'))
    call(['mne_surf2bem --surf %s/surf/lh.seghead ' % op.join(fs_subjects_dir, subject) +
          '--id 4 --force --fif %s-head.fif'  % subject], shell=True, env=os.environ)
    os.chdir(subjects_dir)

    call(['mne_setup_mri'], shell=True, env=os.environ)
    print('Coregistration','In this next interactive GUI, will need to\n' +
          '1. Load the pial surface file -> Load Surface -> Select Pial Surface\n' +
          '2. Load the subject\'s digitization data: File -> Load digitizer data ->' +
          ' Select the raw data file for this session\n' + 
          '3. Open the coordinate alignment window: Adjust -> Coordinate Alignment\n' + 
          '4. Open the viewer window: View -> Show viewer\n' +
          '5. In the coordinate alignment window, click RAP, LAP and Naision, ' + 
          'and then after clicking each of those click on the corresponing ' +
          'fiducial points on the reconstructed head model\n' +
          '6. Click align using fiducials\n' +
          '7. In the View window: select Options -> Show digitizer data\n'
          '8. Adjust the x, y and z coordinates and rotation until ' +
          'the alignment is as close to the ground truth as possible.')
    call(['mne_analyze'], shell=True, env=os.environ)
    os.chdir(this_dir)


def BIDS2MEEGbuddies(bids_dir, out_dir, fs_subjects_dir=None, recon=True, 
                     ico=4, conductivity=(0.3, 0.006, 0.3), 
                     spacing='oct6', surface='white', overwrite=False):
    from subprocess import call
    from mne_bids import read_raw_bids
    from mne_bids.utils import _parse_bids_filename, _find_matching_sidecar
    from mne import pick_types
    from pandas import read_csv
    if recon:
        source_fs_and_mne(fs_subjects_dir)
        ico2 = spacing.replace('oct', '-').replace('ico', '-')
    
    this_dir = os.getcwd()
    df = read_csv(op.join(bids_dir, 'participants.tsv'), sep='\t')
    subjects = [sub.replace('sub-', '') for sub in df['participant_id']]
    MEEGbuddies = []
    for subject in subjects:
        bids_fnames = (glob.glob(op.join(bids_dir, 'sub-%s' % subject, '*', '*',
                                        'sub-%s*.fif' % subject)) + 
                       glob.glob(op.join(bids_dir, 'sub-%s' % subject, '*', '*',
                                        'sub-%s*.vhdr' % subject)))
        for bids_fname in bids_fnames:
            raw = read_raw_bids(op.basename(bids_fname), bids_dir)
            params = _parse_bids_filename(op.basename(bids_fname), True)
            events_fname = _find_matching_sidecar(op.basename(bids_fname), bids_dir, 'events.tsv',
                                                  allow_fail=True)
            df = read_csv(events_fname, sep='\t')
            if op.isfile(op.join(bids_dir, 'task-%s_events.json' % params['task'])):
                with open(op.join(bids_dir, 'task-%s_events.json' % params['task']), 'r') as f:
                    behavior_description = json.load(f)
            else: 
                behavior_description = {col: 'Please describe column here' for col in df.columns}
            no_response = ([i for i, rt in enumerate(df['Response Time']) if np.isnan(rt) or rt == 99]
                           if 'Response Time' in df else None)
            descriptionf = op.join(bids_dir, 'dataset_description.json')
            if not op.isfile(descriptionf):
                raise ValueError('No dataset description file')
            with open(descriptionf, 'r') as f:
                description = json.load(f)
            if params['task'] not in description:
                raise ValueError('The baseline trigger, stimulus triggers and ' +
                                 'response triggers are not defined in the ' +
                                 'dataset_description.json file. Please edit this to' +
                                 'include this information to continue')
            task_triggers = description[params['task']]
            fdata = op.join(out_dir, subject, op.basename(bids_fname))
            if not op.isdir(op.dirname(fdata)):
                os.makedirs(op.dirname(fdata))
            raw.save(fdata, verbose=False, overwrite=True)
            if recon:
                recon_subject(subject, bids_dir, out_dir, fs_subjects_dir, overwrite,
                              ico, ico2, conductivity, spacing, surface)
            meg = pick_types(raw.info, meg=True, eeg=False).size > 0
            eeg = pick_types(raw.info, meg=False, eeg=True).size > 0 
            ecog = pick_types(raw.info, meg=False, ecog=True).size > 0 
            seeg = pick_types(raw.info, meg=False, seeg=True).size > 0 
            MEEGbuddies.append(MEEGBuddy(subject=subject, session=params['ses'], run=params['run'],
                                         fdata=fdata, behavior=events_fname, 
                                         baseline=task_triggers['Baseline'], 
                                         stimuli=task_triggers['Stimuli'],
                                         response=task_triggers['Response'],
                                         task=params['task'], meg=meg, eeg=eeg, ecog=ecog, 
                                         seeg=seeg, subjects_dir=out_dir,
                                         no_response=no_response,
                                         fs_subjects_dir=fs_subjects_dir, 
                                         behavior_description=behavior_description))
    return MEEGbuddies

