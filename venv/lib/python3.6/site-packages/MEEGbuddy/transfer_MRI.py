import numpy as np
import os,sys
from subprocess import call
from tqdm import tqdm

''' Note: Source Freesurfer and MNE before running '''

def transfer_MRI(subject,fname,target,tasks):
    target_dir = target + subject.upper() + '_RAW/'
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)
    command = ['unpacksdcmdir -src %s -targ %s -scanonly %sscan.info'
                %(fname,target_dir,target_dir)]
    scanf = '%sscan.info' %(target_dir)
    if os.path.isfile(scanf):
        print('Scans already imported')
    else:
        call(command, env=os.environ,shell=True)

    while not os.path.isfile(scanf):
        pass

    cfg = '%scfg.txt' %(target_dir)
    i = 1
    with open(cfg,'w') as f:
        with open(scanf,'r') as rf:
            name_indices = {}
            for line in rf:
                index,name,ok,_,_,_,_,dcmname = line.rstrip().split()
                if name in name_indices:
                    name_indices[name] += 1
                else:
                    name_indices[name] = 1
                for task in tasks:
                    if task in name:
                        f.write('%i\t%s\tDICOM\t%s_%s_%i.dcm\t\n'
                                %(i,name,subject.upper(),'BOLD',name_indices[name]))
                if 'FLASH' in name:
                    f.write('%i\t%s\tDICOM\t%s_%s_%i.dcm\t\n'
                            %(i,'FLASH',subject.upper(),name,name_indices[name]))
                elif 'DTI' in name:
                    f.write('%i\t%s\tDICOM\t%s_%s_%i.dcm\t\n'
                            %(i,'FLASH',subject.upper(),name,name_indices[name]))
                else:
                    f.write('%i\t%s\tDICOM\t%s_%s_%i.dcm\t\n'
                            %(i,name,subject.upper(),name,name_indices[name]))
                i += 1

    command = ['unpacksdcmdir -src %s -targ %s -cfg %s' %(fname,target_dir,cfg)]
    call(command, env=os.environ,shell=True)

    for d in os.listdir(target_dir):
        if 'MEMPRAGE' in d:
            targ = max([int(subd) for subd in os.listdir(target_dir+d) if subd.isdigit()])
            t1name = os.listdir('%s%s/%0.3d' %(target_dir,d,targ))[1]
            call(['mri_convert %s%s/%0.3d/%s %sT1.mgz' %(target_dir,d,targ,t1name,target_dir)],
                 env=os.environ,shell=True)
    if task in d:
        i = 1
        for targ in sorted([int(subd) for subd in os.listdir(target_dir+d) if subd.isdigit()]):
            taskfname = os.listdir('%s%s/%0.3d' %(target_dir,d,targ))[1]
            call(['mri_convert %s%s/%0.3d/%s %s%s_%i.mgz'
                  %(target_dir,d,targ,taskfname,target_dir,task,i)],
                 env=os.environ,shell=True)
            i += 1

    os.putenv('SUBJECT',subject)
    os.putenv('SUBJECTS_DIR',target)

    if not os.path.isdir(os.path.join(target,subject)):
        call(['recon-all -i %sT1.mgz -subjid %s -sd %s' %(target_dir,subject,target)],
             env=os.environ, shell=True)
        call(['recon-all -all -subjid %s -sd %s' %(subject,target)],
             env=os.environ, shell=True)
    #call([('cd /autofs/space/karima_001/users/alex/software/mmvt_root/mmvt/; py3; ' +
    #      'source /usr/local/freesurfer/nmr-stable60-env; mne;')],
    #      env=os.environ, shell=True)
    # DKT40 patch
    if not os.path.isfile('%s%s/label/lh.aparc.DKTatlas40.annot' %(target,subject)):
        call(['cp %s%s/label/lh.aparc.DKTatlas.annot ' %(target,subject) +
              '%s%s/label/lh.aparc.DKTatlas40.annot' %(target,subject)],
              env=os.environ, shell=True)
    if not os.path.isfile('%s%s/label/rh.aparc.DKTatlas40.annot' %(target,subject)):
        call(['cp %s%s/label/rh.aparc.DKTatlas.annot ' %(target,subject) +
              '%s%s/label/rh.aparc.DKTatlas40.annot' %(target,subject)],
              env=os.environ, shell=True)
    #call(['python -m src.preproc.anatomy -s %s --remote_subject_dir %s' %(subject,target)],
    #      env=os.environ, shell=True)
    '''if (not os.path.isdir('%s%s/mri/T1-neuromag' %(target,subject)) or not
        os.path.isdir('%s%s/bem/flash'%(target,subject))):
        call(['./set_up_source_space'],env=os.environ, shell=True)'''


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Input subject code, file name on Bourget')
    else:
        _,subject,fname = sys.argv
        transfer_MRI(subject,fname)