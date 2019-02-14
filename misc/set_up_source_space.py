import os,sys
from subprocess import call

def setup_source_space(target,subject):
    if not os.path.isfile('%s%s/bem/%s-oct-6p-src.fif' %(target,subject,subject)):
        call(['mne_setup_source_space --ico -6 --cps --overwrite'],
             env=os.environ,shell=True)

    if not os.path.isdir('%s%s/flash05' %(target,subject)):
        os.makedirs('%s%s/flash05' %(target,subject))
        flash_dir = [d for d in os.listdir('%s%s_RAW/FLASH/' %(target,subject.upper()))
                     if os.path.isdir('%s%s_RAW/FLASH/%s' %(target,subject.upper(),d))]
        if len(flash_dir) > 1:
            print('Warning: Using last FLASH scan found')
        flash_dir = os.path.join(target,subject.upper()+'_RAW','FLASH',flash_dir[-1])
        os.chdir('%s%s/flash05' %(target,subject))
        call(['mne_organize_dicom %s' %(flash_dir)], env=os.environ,shell=True)
        new_flash_dir = os.path.join(target,subject,'flash05',os.path.basename(flash_dir)
                                     +'_MEFLASH_8e_05deg')
        call(['ln -s %s flash05' %(new_flash_dir)], env=os.environ,shell=True)
        call(['mne_flash_bem --noflash30'], env=os.environ,shell=True)
        for area in ['inner_skull','outer_skull','outer_skin']:
            link = os.path.join(target,subject,'bem','%s.surf')
            flash_link = os.path.join(target,subject,'bem','flash','%s.surf')
            if os.path.isfile(link %(area)):
                os.remove(link %(area))
            call(['ln -s %s %s' %(flash_link %(area),link %(area))], env=os.environ,
                 shell=True)
            call(['mne_setup_forward_model --subject %s --surf --ico 4 --innershift 2' %(subject)],
                 env=os.environ,shell=True)
        os.chdir(target)

    # make head model
    if not os.path.isfile(os.path.join(target,subject,'mri','seghead.mgz')):
        call(['mkheadsurf -subjid %s' %(subject)],env=os.environ,shell=True)
        os.chdir(os.path.join(target,subject,'bem'))
        call(['mne_surf2bem --surf %s%s/surf/lh.seghead --id 4 --check --fif %s-head.fif'
              %(target,subject,subject)],env=os.environ,shell=True)
        call(['mne_setup_mri'],env=os.environ,shell=True)
        os.chdir(target)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Enter subject_dir and subject ')
    else:
        setup_source_space(sys.argv[1],sys.argv[2])