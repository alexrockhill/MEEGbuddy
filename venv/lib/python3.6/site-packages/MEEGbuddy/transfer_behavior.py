import glob
from pandas import DataFrame
from scipy.io import loadmat
import numpy as np
import os,sys
import re

def transfer_behavior(task):
    subjectfiles = glob.glob(os.getcwd() + '/DATA/*.mat')
    subjectfiles = [f for f in subjectfiles if not 'params' in f]
    rex_subject = re.compile(r'[a-z]{2}\d{3}')
    rex_block = re.compile(r'block_\d{1}')
    rex_date = re.compile(r'20[0-9]{2}_\d{2}_\d{2}')

    def parsemat(f):
        fmat = loadmat(f,squeeze_me=True,struct_as_record=False)
        TrialStruct = fmat['TrialStruct']
        Trials = TrialStruct.Trials
        ddict = {field:[] for field in Trials[0]._fieldnames}
        for trial in Trials:
            for field in trial._fieldnames:
                try:
                    ddict[field].append(float(getattr(trial,field)))
                except:
                    ddict[field].append(str(getattr(trial,field)))
                    #ddict[field].append(np.nan)
        d = DataFrame(ddict)
        d = d[d.ResponseType != '[]']
        return d

    dfs = {}
    for f in subjectfiles:
        subject = rex_subject.search(f)
        block = rex_block.search(f)
        date = rex_date.search(f)
        if all([subject,block,date]):
            subject = subject.group()
            date = date.group()
            if 'eeg' in f:
                modality = 'eeg'
            elif 'mri' in f:
                modality = 'mri'
            if 'tt' in subject or 'ts' in subject:
                continue
            d = parsemat(f)
            if subject in dfs:
                if modality in dfs[subject]:
                    if date in dfs[subject][modality]:
                        dfs[subject][modality][date] = \
                            dfs[subject][modality][date].append(d).sort_values('Trial')
                    else:
                        dfs[subject][modality][date] = d
                else:
                    dfs[subject][modality] = {date:d}
            else:
                dfs[subject] = {modality:{date:d}}

    for subject in dfs:
        dirname = os.getcwd() + '/%s/behavior/' %(subject)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        for modality in dfs[subject]:
            for i,date in enumerate(dfs[subject][modality]):
                fname = '%s%s_%s_%s_%i' %(dirname,subject,task,modality,i+1)
                df = dfs[subject][modality][date]
                #if not os.path.isfile(fname):
                df.to_csv(fname, index=False)

if __name__ == '__main__':
    transfer_behavior(sys.argv[1])
