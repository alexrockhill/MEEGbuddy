import subprocess
import os.path as op
import os,sys,mne
from tkinter import messagebox
from tkinter import simpledialog
from tkinter.filedialog import askopenfilename,askdirectory
from pathlib import Path
import time
import numpy as np
from tkinter import (Tk, Canvas, Frame, Label, Button, Entry, Listbox, Scrollbar,
                     OptionMenu, StringVar, IntVar, Checkbutton, Text)
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json
from MEEGbuddy import MEEGbuddy
from mne import create_info, find_events, pick_types
from mne.io import Raw, read_raw_brainvision, RawArray
from mne.channels import read_dig_montage

class TMS_EEG_GUI(Frame):

    data = None
    buddy = None

    def __init__(self, root):
        self.root = root
        self.root.title('TMS-EEG GUI')
        self.init_data_dir()
        width = self.root.winfo_screenwidth()
        height = self.root.winfo_screenheight()
        self.size = min([height,width])*0.9
        self.large_font = ('Helvetica',int(self.size*24/1080))
        self.medium_font = ('Helvetica',int(self.size*14/1080))
        self.small_font = ('Helvetica',int(self.size*10/1080))
        self.default_tmin = -2.0
        self.default_tmax = 2.0
        self.default_bl_tmin = -1.1
        self.default_bl_tmax = -0.1

        Frame.__init__(self, root, width=self.size, height=self.size)
        self.pack()
        self.recon_screen()
        #self.load_or_init_screen()
        #self.trials_rejection_screen()

    def label(self,loc,text,font):
        Label(self.getFrame(*loc),text=text,
              wraplength=(loc[3]-loc[2])*self.size,font=font
              ).pack(fill='both',expand=1)


    def entry(self,loc,text,font,default=None):
        entry = Entry(self.getFrame(*loc),font=font)
        entry.pack(fill='both',expand=1)
        entry.focus_set()
        if default is not None:
            entry.insert(0,default)
        self.data_entries[text] = entry
        return entry


    def file_button(self,loc,text,font,ftypes):
        file_button = Button(self.getFrame(*loc),text='Browse',
                                font=font)
        file_button.pack(fill='both',expand=1)
        file_button.configure(command =
            lambda: self.show_file_browser(self.data_entries[text],ftypes))


    def button(self,loc,text,font,command):
        button = Button(self.getFrame(*loc),text=text,font=font)
        button.pack(fill='both',expand=1)
        button.configure(command=self.recon)


    def text(self,loc,text,font):
        output = Text(self.getFrame(*loc),font=font)
        output.pack(fill='both',expand=1)
        self.data_entries[text] = output

    def recon_screen(self):
        self.data_entries = {}
        # Headers
        self.label([0,0.05,0,1],'Freesurfer Reconstruction',self.large_font)
        # File loaders
        self.label([0.05,0.15,0,0.25],'T1 Image Location',self.large_font)
        self.entry([0.05,0.15,0.25,0.75],'T1 Image Location',self.large_font)
        self.file_button([0.05,0.15,0.75,1],'T1 Image Location',self.large_font,
                         (('mgz', '*.mgz'),('nii', '*.nii'),('mgh', '*.mgh'),('dicom', '*')))
        self.label([0.15,0.25,0,0.25],'FLASH Image Location (optional)',self.large_font)
        self.entry([0.15,0.25,0.25,0.75],'FLASH Image Location (optional)',self.large_font)
        self.file_button([0.15,0.25,0.75,1],'FLASH Image Location (optional)',self.large_font,
                         (('mgz', '*.mgz'),('nii', '*.nii'),('mgh', '*.mgh'),('dicom', '*')))

        self.label([0.25,0.35,0,0.25],'Subject',self.large_font)
        self.entry([0.25,0.35,0.25,0.75],'Subject',self.large_font)

        self.label([0.35,0.45,0,0.25],'Subjects Directory',self.large_font)
        self.entry([0.35,0.45,0.25,0.75],'Subjects Directory',self.large_font)
        self.file_button([0.35,0.45,0.75,1],'Subjects Directory',self.large_font,'dir')

        self.label([0.45,0.55,0,0.25],'Boundary Element Model',self.medium_font)
        self.label([0.45,0.50,0.25,0.45],'ico',self.medium_font)
        self.entry([0.50,0.55,0.25,0.45],'ico',self.medium_font,default='4')
        self.label([0.45,0.50,0.45,0.65],'conductivity',self.medium_font)
        self.entry([0.50,0.55,0.45,0.65],'conductivity',self.medium_font,
                   default='(0.3, 0.006, 0.3)')
        self.label([0.45,0.50,0.65,1],'BEM fname',self.medium_font)
        self.entry([0.50,0.55,0.65,1],'BEM fname',self.medium_font,
                   default='-bem-sol.fif')

        self.label([0.55,0.65,0,0.25],'Source Space',self.medium_font)
        self.label([0.55,0.60,0.25,0.45],'spacing',self.medium_font)
        self.entry([0.60,0.65,0.25,0.45],'spacing',self.medium_font,
                   default='oct6')
        self.label([0.55,0.60,0.45,0.65],'surface',self.medium_font)
        self.entry([0.60,0.65,0.45,0.65],'surface',self.medium_font,
                   default='white')
        self.label([0.55,0.60,0.65,1.0],'SRC fname',self.medium_font)
        self.entry([0.60,0.65,0.65,1.0],'SRC fname',self.medium_font,
                   default='-src.fif')

        self.text([0.65,0.85,0.1,0.9],'Command Output',self.medium_font)

        self.button([0.85,0.925,0.45,0.55],'Recon',self.medium_font,self.recon)
        self.button([0.85,0.925,0.85,0.95],'Skip',self.medium_font,
                    self.transition_load_init_screen)


    def recon(self):
        subject = self.data_entries['Subject'].get()
        os.environ['SUBJECT'] = subject
        subjects_dir = self.data_entries['Subjects Directory'].get()
        os.environ['SUBJECTS_DIR'] = subjects_dir
        try:
            self.output_to_terminal('source $FREESURFER_HOME/SetUpFreeSurfer.sh')
        except:
            raise ValueError('Freesurfer not installed or installed correctly. ' + 
                             'Make sure $FREESURFER_HOME is defined correctly.' +
                             'See https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall')
        try:
            shell = op.basename(os.environ['SHELL'])
            if shell == 'bash' or shell == 'tcsh':  # might have to change these only tested for tcsh
                self.output_to_terminal('source $MNE_ROOT/bin/mne_setup_sh')
            elif shell == 'csh':
                self.output_to_terminal('source $MNE_ROOT/bin/mne_setup')
            else:
                raise ValueError('Shell not bash or csh or not understood')
        except:
            raise ValueError('MNE_C not installed or installed correctly. ' +
                             'Make sure $MNE_ROOT is defined correctly. ' +
                             'See https://martinos.org/mne/stable/install_mne_c.html')

        t1 = self.data_entries['T1 Image Location'].get()
        if op.isdir(op.join(subjects_dir,subject,'mri')):
            self.print_to_terminal('Recon already run, skipping')
        else:
            self.output_to_terminal('recon-all -subjid %s ' %(subject) + 
                                    '-i %s --all' %(t1))

        ico = int(self.data_entries['ico'].get())
        conductivity = np.array(self.data_entries['conductivity'].get())
        spacing = self.data_entries['spacing'].get(),
        surface=self.data_entries['surface'].get()

        srcf = op.join(subjects_dir,subject,'src',
                       self.data_entries['SRC fname'].get())
        '''if op.isfile(srcf):
            raise ValueError('Source file already exists, change name or delete original file')
        src = mne.setup_source_space(subject,spacing=spacing,surface=surface,
                                     subjects_dir=subjects_dir,add_dist=False)
        src.save(srcf)'''
        
        self.output_to_terminal('mne_setup_source_space --spacing %s ' %(spacing) +
                                '--ico %i --surf %s --cps ' %(ico,surface) + 
                                '--src %s --overwrite' %(srcf))
        #Organize flash if supplied
        flash = self.data_entries['FLASH Image Location (optional)'].get()
        if flash:
            if op.isdir(op.join(subjects_dir,subject,'/flash05')):
                os.rmdir(op.join(subjects_dir,subject,'/flash05'))
            os.makedirs(op.join(subjects_dir,subject,'/flash05'))
            os.chdir(op.join(subjects_dir,subject,'/flash05'))
            flash_dir = os.dirname(self.data_entries['Flash'].get())
            self.output_to_terminal('mne_organize_dicom %s' %(flash_dir))
            new_flash_dir = op.join(subjects_dir,subject,'flash05',op.basename(flash_dir)
                                         +'_MEFLASH_8e_05deg')
            self.output_to_terminal('ln -s %s flash05' %(new_flash_dir))
            self.output_to_terminal('mne_flash_bem --noflash30')
            self.output_to_terminal('freeview -v %s/%s/mri/T1.mgz ' %(subjects_dir,subject) + 
                '-f %s/%s/bem/flash/inner_skull.surf ' %(subjects_dir,subject)+
                '-f %s/%s/bem/flash/outer_skull.surf ' %(subjects_dir,subject)+
                '-f %s/%s/bem/flash/outer_skin.surf' %(subjects_dir,subject))
            for area in ['inner_skull','outer_skull','outer_skin']:
                link = op.join(subjects_dir,subject,'bem','%s.surf' %(area))
                flash_link = op.join(subjects_dir,subject,'bem','flash',
                                         '%s_%s_surface' %(subject,area))
                if op.isfile(link):
                    os.remove(link)
                self.output_to_terminal('ln -s %s %s' %(flash_link,link))
        else:
            self.output_to_terminal('mne_watershed_bem --subject %s --atlas' %(subject))
            self.output_to_terminal('freeview -v %s/%s/mri/T1.mgz ' %(subjects_dir,subject) + 
                 '-f %s/%s/bem/watershed/%s_inner_skull_surface ' %(subjects_dir,subject,subject) +
                 '-f %s/%s/bem/watershed/%s_outer_skull_surface ' %(subjects_dir,subject,subject)+ 
                 '-f %s/%s/bem/watershed/%s_outer_skin_surface' %(subjects_dir,subject,subject))
            for area in ['inner_skull','outer_skull','outer_skin']:
                link = op.join(subjects_dir,subject,'bem','%s.surf' %(area))
                watershed_link = op.join(subjects_dir,subject,'bem','watershed',
                                         '%s_%s_surface' %(subject,area))
                if op.isfile(link):
                    os.remove(link)
                self.output_to_terminal('ln -s %s %s' %(watershed_link,link))
        # BEM
        # if this fails, here is the work around https://github.com/ezemikulan/blender_freesurfer
        # you also need this https://github.com/andersonwinkler/toolbox for the brainder commands (in toolbox/bin)
        bemf = op.join(subjects_dir,subject,'bem',
                       self.data_entries['BEM fname'].get())
        '''if op.isfile(bemf):
            print('WARNING, BEM file already exists, overwriting original file')
        model = mne.make_bem_model(subject=subject,ico=ico,conductivity=conductivity,
                                   subjects_dir=subjects_dir)
        bem = mne.make_bem_silution(model)
        mne.write_bem_solution(bemf,bem)'''

        self.output_to_terminal('mne_setup_forward_model --subject %s --surf' %(subject) + 
                                '--ico %i --brainc %f --skullc %f --scalpc %f' %(ico,*conductivity) +
                                '--innershift %i --model %s' %(2 if flash else -1,bemf))
        if not op.isfile(bemf):
            raise ValueError('Forward model failed, likely due to errors in ' + 
                             'freesurfer reconstruction and segementation. ' + 
                             'See https://github.com/ezemikulan/blender_freesurfer ' +
                             'for a workaround, functions from ' + 
                             'https://github.com/andersonwinkler/toolbox are needed too')
        os.chdir(subjects_dir)

        # make head model
        self.output_to_terminal('mkheadsurf -subjid %s' %(subject))
        os.chdir(op.join(subjects_dir,subject,'bem'))
        self.output_to_terminal('mne_surf2bem --surf ' +
                                '%s/surf/lh.seghead ' %(op.join(subjects_dir,subject)) +
                                '--id 4 --force --fif %s-head.fif'  %(subject))
        os.chdir(subjects_dir)

        self.output_to_terminal('mne_setup_mri')

        messagebox.showinfo('Coregistration','In this next interactive GUI, will need to\n' +
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
                            'the alignment is as close to the ground truth as possible\n' +
                            'If you don\'t know these instructions, feel free to copy and paste')

        self.output_to_terminal('mne_analyze')


    def output_to_terminal(self,command):
        output = subprocess.Popen([command],env=os.environ,
                                  shell=True,stdout=subprocess.PIPE)
        while True:
            line = output.stdout.readline()
            self.data_entries['Command Output'].insert('end',line)
            if not line: break


    def print_to_terminal(self,text):
        self.data_entries['Command Output'].insert('end','\n' + text)


    def transition_load_init_screen(self):
        self.clear_screen()
        self.load_or_init_screen()

    def init_data_dir(self):
        self.data_dir = './TMS_EEG_GUI_data'
        if not op.isdir(self.data_dir):
            os.makedirs(self.data_dir)
        sub_dirs_names = ['saved_exp','data','tmp']
        self.sub_dirs = {}
        for sub_dir_name in sub_dirs_names:
            sub_dir_full_path = op.join(self.data_dir,sub_dir_name)
            if not op.isdir(sub_dir_full_path):
                os.makedirs(sub_dir_full_path)
            self.sub_dirs[sub_dir_name] = sub_dir_full_path

    def show_file_browser(self,entry,ftypes):
        initialdir = (os.environ['SUBJECTS_DIR'] if 
                      'SUBJECTS_DIR' in os.environ else None)
        if ftypes == 'dir':
            fname = askdirectory(initialdir=initialdir,
                                 title='Select directory')
        else:
            fname = askopenfilename(initialdir=initialdir, 
                                    title='Select file',filetypes=ftypes)
        entry.delete(0,'end')
        entry.insert(0,fname)

    def save_setup(self):
        de = self.data_entries.copy()
        raw_fname = de.pop('Load New File').get()
        loc_fname = de.pop('Load Location File').get()
        suffix = raw_fname.split('.')[-1]
        if suffix == 'fif':
            try:
                raw = Raw(raw_fname,preload=False)
            except Exception as e:
                print(e)
                messagebox.showerror('Error','Failed to read fif')
                return
            if loc_fname:
                messagebox.showwarning('Warning','Ignoring location file given')
        elif suffix == 'vhdr':
            if not loc_fname:
               messagebox.showerror('Error','Location file not specified')
            try:
                raw = read_raw_brainvision(raw_fname,
                    preload=op.join(self.sub_dirs['tmp'],'raw_tmp'))
            except Exception as e:
                print(e)
                messagebox.showerror('Error','Failed to read BrainVision raw file')
                return
            try:
                montage = read_dig_montage(bvct=loc_fname)
                ref_name = [ch for ch in montage.dig_ch_pos if not
                            ch in raw.info['ch_names']]
                if len(ref_name) != 1:
                    ref_name = simpledialog.askstring('Reference Channel',
                                                      'What is the reference channel name?',
                                                      parent=self.root)
                else:
                    ref_name = ref_name[0]
                info = create_info([ref_name], raw.info['sfreq'], ['eeg'])
                ref = RawArray(np.zeros((1, len(raw.times))), info)
                ref.info['lowpass'] = raw.info['lowpass']
                raw = raw.add_channels([ref]) #, force_update_info=True)
                #
                for ch_type in ['eog','ecg','emg','stim']:
                    chs = [ch for ch in raw.info['ch_names']
                           if ch_type in ch.lower()]
                    for ch in chs:
                        ch_ix = raw.ch_names.index(ch)
                        raw._data[ch_ix, :] *= 1e-6
                        raw.set_channel_types({ch:ch_type})
                raw = raw.set_montage(montage,verbose=False)
            except Exception as e:
                print(e)
                messagebox.showerror('Error','Failed to read BrainVision location file')
                return
        else:
            messagebox.showerror('Error','Unrecognized raw file format')
            return
        self.save_exp(raw)

    def save_exp(self,raw):
        self.data = {}
        name_items = ['Name','Condition (e.g. Awake)','Date','Time','Target','Hemisphere']
        fname = '_'.join([str(self.data_entries[i].get()).replace('.','_').replace('/','_')
                             for i in name_items])
        if not op.isdir(op.join(self.sub_dirs['data'],'raw')):
            os.makedirs(op.join(self.sub_dirs['data'],'raw'))
        exp_fname = op.join(self.sub_dirs['saved_exp'],fname+'.json')
        if op.isfile(exp_fname):
            ok = messagebox.askokcancel('TMS-EEG GUI','Overwrite previous data?')
            if not ok:
                return
        stim_inds = pick_types(raw.info,meg=False,stim=True)
        stim_chs = [raw.ch_names[ind] for ind in stim_inds]
        triggers = []
        for stim_ch in stim_chs:
            print(stim_ch)
            try:
                events = find_events(raw,stim_channel=stim_ch)
            except Exception as e:
                print(e)
                messagebox.showerror('Error','Error reading triggers')
                return
            this_triggers = np.unique(events[:,2])
            triggers += list(this_triggers)
            info = create_info([str(t) for t in this_triggers],
                               raw.info['sfreq'], ['stim' for _ in this_triggers])
            arr = np.zeros((len(this_triggers), len(raw.times)))
            j = raw.first_samp + 1
            for k,this_trigger in enumerate(this_triggers):
                this_events = events[np.where(events[:,2]==this_trigger)[0]]
                for i in this_events[:, 0]:
                   arr[k, i-j : i-j+100] = 1
            chs = RawArray(arr,info,verbose=False)
            chs.info['lowpass'] = raw.info['lowpass']
            raw.add_channels([chs])
        raw.drop_channels(stim_chs)
        raw_fname = op.join(self.sub_dirs['data'],'raw',fname + '-raw.fif')
        raw.save(raw_fname, overwrite=True)
        self.data['fdata'] = raw_fname
        if len(triggers) == 0:
            messagebox.showerror('Error','No triggers found')
            return
        triggers = np.unique(events[:,2])

        self.data['triggers'] = ','.join([str(e) for e in triggers])
        self.data['n_epochs'] = len(np.where(events[:,2] == triggers[0])[0])
        for key in self.data_entries:
            self.data[key] = self.data_entries[key].get()
        with open(exp_fname,'w') as f:
            json.dump(self.data,f)


    def load_exp(self,fname):
        with open(op.join(self.sub_dirs['saved_exp'],fname),'r') as f:
            self.data = json.load(f)
            for key in self.data:
                if key in self.data_entries:
                    self.data_entries[key].delete(0,'end')
                    self.data_entries[key].insert(0,self.data[key])


    def getFrame(self,y0,y1,x0,x1):
        frame = Frame(self.root,
                      height=(y1-y0)*self.size-self.size/100,
                      width=(x1-x0)*self.size-self.size/100)
        frame.pack_propagate(0)
        frame.place(x=x0*self.size,y=y0*self.size)
        return frame


    def load_or_init_screen(self):
        self.data_entries = {}
        # Headers
        self.label([0,0.1,0,0.5],'Create New',self.large_font)
        self.label([0,0.1,0.5,1],'Load',self.large_font)
        # File loaders
        self.label([0.1,0.15,0,0.1],'Load New File',self.medium_font)
        self.entry([0.1,0.15,0.1,0.4],'Load New File',self.medium_font)

        self.label([0.15,0.2,0,0.1],'Load Location File',self.medium_font)
        self.entry([0.15,0.2,0,0.1],'Load Location File',self.medium_font)

        load_new_file_button = Button(self.getFrame(0.1,0.15,0.4,0.5),text='Browse',
                                      font=self.medium_font)
        load_new_file_button.pack(fill='both',expand=1)
        load_new_file_button.configure(command =
            lambda: self.show_file_browser(new_file,(('vhdr', '*.vhdr'),
                                                     ('fif', '*.fif'))))
        load_loc_file_button = Button(self.getFrame(0.15,0.2,0.4,0.5),text='Browse',
                                      font=self.medium_font)
        load_loc_file_button.pack(fill='both',expand=1)
        load_loc_file_button.configure(command =
            lambda: self.show_file_browser(loc_file,(('bvct', '*.bvct'),)))
        Label(self.getFrame(0.1,0.2,0.5,0.6),text='Load previous',wraplength=0.1*self.size,
              font=self.medium_font).pack(fill='both',expand=1)
        scrollbar_v = Scrollbar(self.getFrame(0.1,0.18,0.88,0.9), orient='vertical')
        scrollbar_h = Scrollbar(self.getFrame(0.18,0.2,0.6,0.88), orient='horizontal')
        load_file_list = Listbox(self.getFrame(0.1,0.18,0.6,0.88), height=3,
                                 font=self.large_font,
                                 yscrollcommand=scrollbar_v.set,
                                 xscrollcommand=scrollbar_h.set)
        load_file_list.pack(fill='both',expand=1)
        load_files = [f for f in os.listdir(self.sub_dirs['saved_exp'])
                      if f.split('.')[-1] == 'json']
        for lf in [''] + load_files:
            load_file_list.insert('end',lf)
        scrollbar_v.config(command=load_file_list.yview)
        scrollbar_v.pack(fill='both',expand=1)
        scrollbar_h.config(command=load_file_list.xview)
        scrollbar_h.pack(fill='both',expand=1)
        load_previous_button = Button(self.getFrame(0.1,0.2,0.9,1),text='Load',
                                      font=self.medium_font)
        load_previous_button.pack(fill='both',expand=1)
        load_previous_button.configure(command =
            lambda: self.load_exp(load_file_list.get(load_file_list.curselection())))
        # experiment data
        def make_data_label(y0,y1,text,browse=False):
            Label(self.getFrame(y0,y1,0,0.2),text=text,
                  font=self.medium_font).pack(fill='both',expand=1)
            entry = Entry(self.getFrame(y0,y1,0.2,0.9),font=self.large_font)
            entry.pack(fill='both',expand=1)
            entry.focus_set()
            if browse is not None:
                load_button = Button(self.getFrame(y0,y1,0.9,1.0),text='Browse',
                                     font=self.large_font)
                load_button.pack(fill='both',expand=1)
                load_button.configure(command =
                    lambda: self.show_file_browser(new_file,browse))
            self.data_entries[text] = entry
            return entry
        categories = ['Name','Condition','Date','Time','Target','Hemisphere',
                      'Intensity','RMT','Description','FS Dir','BEM File',
                      'Source File','Coord Transf File']
        browse_dict = {'FS Dir':'dir','BEM File':(('bemf','*-bem-sol.fif'),),
                       'Source File':(('src','*-src.fif'),),
                       'Coord Transf File':(('COR','*-COR-fif'),)}
        for i,text in enumerate(categories):
            y0 = 0.2 + i * 0.7/(len(categories)+1)
            y1 = 0.2 + (i+1) * 0.7/(len(categories)+1)
            make_data_label(y0,y1,text,browse=(browse_dict[text] if 
                                               text in browse_dict else None))
        save_button = Button(self.getFrame(0.85,0.925,0.45,0.55),text='Save',
                                      font=self.medium_font)
        save_button.pack(fill='both',expand=1)
        save_button.configure(command=self.save_setup)
        next_button = Button(self.getFrame(0.85,0.925,0.85,0.95),text='Next',
                                      font=self.medium_font)
        next_button.pack(fill='both',expand=1)
        next_button.configure(command=self.transition_trials_rejection)

    def transition_trials_rejection(self):
        if not self.data:
            messagebox.showerror('Error','Experiment configuration not saved/loaded')
            return
        self.init_buddy()
        self.clear_screen()
        self.trials_rejection_screen()

    def make_option_menu(self,frame,options):
        choice = StringVar(frame)
        choice.set(options[0])
        options = OptionMenu(frame,choice,*options)
        options.pack(fill='both',expand=1)
        return choice

    def trials_rejection_screen(self):
        self.data_entries = {}
        Label(self.getFrame(0,0.1,0,0.3),text='Raw Data & Reject Trials',
              font=self.medium_font).pack(fill='both',expand=1)
        Label(self.getFrame(0.1,0.15,0.02,0.08),text='Trigger Number',
              wraplength=0.05*self.size,font=self.small_font
              ).pack(fill='both',expand=1)
        trigger = self.make_option_menu(self.getFrame(0.1,0.15,0.08,0.15),
                                        self.data['triggers'].split(','))
        Label(self.getFrame(0.15,0.2,0.02,0.08),text='Trigger Delay (ms)',
              wraplength=0.05*self.size,font=self.small_font
              ).pack(fill='both',expand=1)
        delay = Entry(self.getFrame(0.15,0.2,0.08,0.15),font=self.small_font)
        delay.pack(fill='both',expand=1)
        delay.focus_set()
        delay.insert(0,'0')
        self.data_entries['Delay'] = delay
        Label(self.getFrame(0.2,0.25,0.02,0.08),text='EOG Threshold',
              wraplength=0.05*self.size,font=self.small_font
              ).pack(fill='both',expand=1)
        EOG_threshold = self.make_option_menu(self.getFrame(0.2,0.25,0.08,0.15),
                                              ['70'])
        self.data_entries['EOG Threshold'] = EOG_threshold
        Label(self.getFrame(0.1,0.15,0.16,0.22),text='Threshold',
              wraplength=0.05*self.size,font=self.small_font
              ).pack(fill='both',expand=1)
        threshold = self.make_option_menu(self.getFrame(0.1,0.15,0.22,0.3),
                                          ['1000'])
        self.data_entries['Threshold'] = threshold
        Label(self.getFrame(0.15,0.17,0.16,0.22),text='Baseline',
              wraplength=0.06*self.size,font=self.small_font
              ).pack(fill='both',expand=1)
        Label(self.getFrame(0.15,0.17,0.22,0.26),text='min',
              wraplength=0.04*self.size,font=self.small_font
              ).pack(fill='both',expand=1)
        Label(self.getFrame(0.15,0.17,0.26,0.3),text='max',
              wraplength=0.04*self.size,font=self.small_font
              ).pack(fill='both',expand=1)
        bl_min = Entry(self.getFrame(0.17,0.2,0.22,0.26),font=self.small_font)
        bl_min.pack(fill='both',expand=1)
        bl_min.focus_set()
        bl_min.insert(0,str(int(self.default_bl_tmin*100)))
        bl_max = Entry(self.getFrame(0.17,0.2,0.26,0.3),font=self.small_font)
        bl_max.pack(fill='both',expand=1)
        bl_max.focus_set()
        bl_max.insert(0,str(int(self.default_bl_tmax*100)))
        self.data_entries['Baseline Min'] = bl_min
        self.data_entries['Baseline Max'] = bl_max
        bl_normalize = IntVar()
        bl_normalize.set(1)
        bl_normalize_radio = Checkbutton(self.getFrame(0.18,0.2,0.16,0.22),
                                         text='Norm',variable=bl_normalize,
                                         command=self.changeNormalize)
        bl_normalize_radio.pack(fill='both',expand=1)
        self.data_entries['Baseline Normalize'] = bl_normalize
        Label(self.getFrame(0.2,0.25,0.16,0.22),text='Interval (ms)',
              wraplength=0.05*self.size,font=self.small_font
              ).pack(fill='both',expand=1)
        Label(self.getFrame(0.2,0.22,0.22,0.26),text='min',
              wraplength=0.04*self.size,font=self.small_font
              ).pack(fill='both',expand=1)
        Label(self.getFrame(0.2,0.22,0.26,0.3),text='max',
              wraplength=0.04*self.size,font=self.small_font
              ).pack(fill='both',expand=1)
        interval_min = Entry(self.getFrame(0.22,0.25,0.22,0.26),font=self.small_font)
        interval_min.pack(fill='both',expand=1)
        interval_min.focus_set()
        interval_min.insert(0,'-600')
        interval_max = Entry(self.getFrame(0.22,0.25,0.26,0.3),font=self.small_font)
        interval_max.pack(fill='both',expand=1)
        interval_max.focus_set()
        interval_max.insert(0,'600')
        self.data_entries['Interval Min'] = interval_min
        self.data_entries['Interval Max'] = interval_max
        epo = self.buddy._load_epochs('TMS')
        fig = epo.plot(n_epochs=1)
        epo_selection = FigureCanvasTkAgg(fig,self.getFrame(0.1,0.9,0.3,0.9))
        epo_selection.draw()
        epo_selection.get_tk_widget().pack(fill='both',expand=1)
        self.data_entries['Epoch Selection'] = epo_selection

    def changeNormalize(self):
        if self.data_entries['Baseline Normalize'].get():
            self.buddy.events['Baseline'] = \
                [self.data['stim_chs'][0],
                 float(self.data_entries['Baseline Min'].get())/100,
                 float(self.data_entries['Baseline Max'].get())/100]
        else:
            self.buddy.events['Baseline'] = None


    def init_buddy(self):
        self.buddy = MEEGbuddy(subject=self.data['Name'],
                               session=self.data['Session'],
                               fdata=self.data['fdata'],
                               behavior=self.data['Behavior'],
                               baseline=[self.data['triggers'][0],
                                                       self.default_bl_tmin,
                                                       self.default_bl_tmax],
                               stimuli={'TMS':[self.data['triggers'][0],
                                               self.default_tmin,
                                               self.default_bl_tmax]},
                               task='TMS-EEG',eeg=True,
                               subjects_dir=self.data['Subjects Directory'],
                               fs_subjects_dir=self.data['FS Dir'],
                               bemf=self.data['BEM File'],
                               srcf=self.data['Source File'],
                               transf=self.data['Coord Transf File'])
        self.buddy.makeEpochs()

    def clear_screen(self):
        for widget in self.root.winfo_children():
            widget.destroy()



if __name__ == '__main__':
    root = Tk()
    gui = TMS_EEG_GUI(root)
    root.mainloop()