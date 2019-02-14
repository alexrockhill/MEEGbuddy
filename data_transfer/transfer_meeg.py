from subprocess import call
import pandas as pd
import os, re, sys
import numpy as np
from mne.io import Raw
from tqdm import tqdm

def transfer(data_type='MEG', tasks=['MSIT','ECR']):

    rex = re.compile(r'[a-z]{2}\d{3}')

    # Transfers the MEEG data into this same directory under
    # meeg subfolders. This is the extent to which the BIDS extension
    # is followed since it is still under development.
    # Must be exectued from this same folder.

    CH_MAPPINGS = {'EEG001': 'Fp1', 'EEG002': 'Fpz', 'EEG003': 'Fp2',
                   'EEG004': 'AF7', 'EEG005': 'AF3', 'EEG006': 'AFz',
                   'EEG007': 'AF4', 'EEG008': 'AF8', 'EEG009': 'F7',
                   'EEG010': 'F5', 'EEG011': 'F3', 'EEG012': 'F1', 'EEG013': 'Fz',
                   'EEG014': 'F2', 'EEG015': 'F4', 'EEG016': 'F6',
                   'EEG017': 'F8', 'EEG018': 'FT9', 'EEG019': 'FT7',
                   'EEG020': 'FC5', 'EEG021': 'FC3', 'EEG022': 'FC1',
                   'EEG023': 'FCz', 'EEG024': 'FC2', 'EEG025': 'FC4',
                   'EEG026': 'FC6', 'EEG027': 'FT8', 'EEG028': 'FT10',
                   'EEG029': 'T9', 'EEG030': 'T7', 'EEG031': 'C5', 'EEG032': 'C3',
                   'EEG033': 'C1', 'EEG034': 'Cz', 'EEG035': 'C2', 'EEG036': 'C4',
                   'EEG037': 'C6', 'EEG038': 'T8', 'EEG039': 'T10',
                   'EEG040': 'TP9', 'EEG041': 'TP7', 'EEG042': 'CP5',
                   'EEG043': 'CP3', 'EEG044': 'CP1', 'EEG045': 'CPz',
                   'EEG046': 'CP2', 'EEG047': 'CP4', 'EEG048': 'CP6',
                   'EEG049': 'TP8', 'EEG050': 'TP10', 'EEG051': 'P9',
                   'EEG052': 'P7', 'EEG053': 'P5', 'EEG054': 'P3',
                   'EEG055': 'P1', 'EEG056': 'Pz', 'EEG057': 'P2', 'EEG058': 'P4',
                   'EEG059': 'P6', 'EEG060': 'P8', 'EOG061': 'HEOG',
                   'EOG062': 'VEOG', 'ECG063': 'ECG', 'EEG065': 'P10',
                   'EEG066': 'PO7', 'EEG067': 'PO3', 'EEG068': 'P0z',
                   'EEG069': 'PO4', 'EEG070': 'PO8', 'EEG071': 'O1',
                   'EEG072': 'Oz', 'EEG073': 'O2', 'EEG074': 'Iz'}

    # DARPA database path
    data_dir = '/space/lilli/3/users/DARPA-TRANSFER/meg/'
    out_dir = os.getcwd() + '/data/'

    for task in tasks:
        print('transferring task %s' %(task))

        info = pd.read_csv('meg_subjects.csv')
        subjects = list(info.loc[info[task], 'Subject'])
        subjects = [s for s in subjects if rex.search(s)]

        for subject in tqdm(subjects):
            # Create the subject directory
            if not os.path.isdir(data_dir + subject):
                print('Subject %s data not found' %(subject))
                continue
            dname = '%s/data/sub-%s/%s/' %(os.getcwd(),subject,data_type.lower())

            hdr = ["ScanNumber", "Subject", "Day", "Block", "Task", "Fname"]
            cfg_info = pd.read_table("%s/%s/cfg.txt" % (data_dir, subject), sep=" ",
                                 header=None, names=hdr)
            task_info = cfg_info[cfg_info.Task == task.lower()]

            for i in task_info.index:
                b_in_file = ('%s%s/%03d/%s_%s_behavior.csv'
                             %(data_dir,subject,task_info.loc[i]['ScanNumber'],
                              subject,task.lower()))
                out_data_dir = ('%ssub-%s/%i/%s/'
                                %(out_dir,subject,task_info.loc[i]['Day'].item(),
                                  data_type.lower()))
                b_out_file = ('%s_%i.csv'
                              %(task.lower(),task_info.loc[i]['Block'].item()))
                if not os.path.isdir(out_data_dir):
                    os.makedirs(out_data_dir)
                call(['cp %s %s' %(b_in_file, out_data_dir + b_out_file)],
                      env=os.environ,shell=True)
                out_file = ('%s/sub-%s/%i/%s/%s_%iraw.fif'
                            %(out_dir,subject,task_info.loc[i]['Day'].item(),
                              data_type.lower(),task,task_info.loc[i]['Block'].item()))
                in_file = ('%s/%s/%03d/%s'
                           %(data_dir, subject, task_info.loc[i]['ScanNumber'],
                             task_info.loc[i]['Fname']))

                # Load in the Raw Data
                raw = Raw(in_file, verbose=False, preload=True)

                # This code just replaces the EEG channel location information
                # with the EEG channel digitization location information
                # Fixes a bug where some occipital channels didn't have the correct
                # location info

                if data_type == 'EEG':
                    # Get digitization information
                    dig = raw.info['dig']

                    # Get coordinates for EEG electrodes from digitization
                    # We ignore the first channel (reference)
                    # and the last four channels (EOG, ).
                    dig_loc = np.array([d['r'] for d in dig
                                        if (d['kind'] == 3 and d['ident'] != 0 and
                                            d['ident'] <= 70)])

                    # Get raw channel information and pick out EEG channels
                    chs = raw.info['chs']
                    eeg_chs = [c for c in chs if 'EEG' in c['ch_name']]

                    # Update channel location information
                    # with digitization location information.
                    for ch, dl in zip(eeg_chs, dig_loc):
                        ch['loc'][: 3] = dl
                    raw.info['chs'] = this_chs

                # Select out m/eeg
                raw.pick_types(eeg=data_type=='EEG',meg=data_type=='MEG',
                               stim=True, eog=True, ecg=True)

                # Remove projections
                raw.info['proj'] = []

                # Rename the channels
                if data_type == 'EEG':
                    raw.rename_channels(CH_MAPPINGS)

                # save out raw
                raw.save(out_file, overwrite=True)

if __name__ == '__main__':
    data_type = sys.argv[1]
    tasks = sys.argv[2:]
    transfer(data_type,tasks)