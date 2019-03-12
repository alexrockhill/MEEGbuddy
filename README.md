# MEEGbuddy

I work in a lab that analyzes many psychphysical tasks with the same structure: there are trials with stimuli and responses. If this matches the description of what you're studying, this tool will probably be very helpful. Using MEEGbuddy, I was able to preprocess ~70 subjects for three tasks using both MEG and EEG in three days not counting computation time.

MEEGbuddy uses MNE, autoreject and PCI (noreun on github) for preprocessing mostly but also analysis of MEEG data. TFR psd multitapers, including sleep scoring, and morlet are supported as well as cluster permutation tests and lempel-ziv complexity (connectivity coming soon hopefully). These preprocessing and analysis steps all come with defaults that will run well as is and have lots of the kinks figured out already with the added bonus that they are saved in BIDS structure (or pretty close).

![alt text](https://raw.githubusercontent.com/alexrockhill/MEEGbuddy/master/MEEGbuddyDemo/data/plots/source_bootstrap/source_bootstrap.jpeg)

![alt text](https://raw.githubusercontent.com/alexrockhill/MEEGbuddy/master/MEEGbuddyDemo/data/plots/source_plot/sample_AudioVis_eeg_meg_Cue_Stimulus_Type_Left_Visual_Autoreject_both_lat_med_cau_dor_ven_fro_par.gif)


## Installation Instructions
First install anaconda (recommended) (https://www.anaconda.com/distribution/#download-section) or your manage your own python (https://www.python.org/downloads/). For this to work properly, you have to be able to run "python" and "pip" from a terminal. To have a terminal recognize python, if it doesn't automatically after install, you have to add python to the path. In MacOS or Linux you can modify the .bash_profile/.bashrc (for terminals with $) or the .cshrc (terminals that end with [1]) by adding "alias python='python3' " and "alias pip='pip3' " if you used anaconda or "alias python='/path/to/bin/python3' " and "alias pip='/path/to/bin/pip3' " (Note: you have to restart the terminal for these changes to take effect). For Windows, it's a little tricker. Find the "Environment Variables" settings window, I would recommend just searching but it should be under Properties>System Properties>Advanced. Then add to the path *both* the /path/to/python and the /path/to/pip which likely is a subdirectory in the python folder (subdirectories do not get added automatically).  
(Note: run "..." means type "..." into a terminal)

0. Optional preinstallation for making a virtual environment so you don't break dependancies in other python programs you use
0a. run "pip install virtualenv --user"
0b. run "virtualenv /Path/to/where/you/want/to/store/your/venv"
0c. run "source /Path/to/where/you/want/to/store/your/venv/bin/activate"
1. run "pip install MEEGbuddy,numpy"
2. run "pip install -r requirements.txt"
3. (Optional) Install freesurfer (https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall) 
4. (Optional) Install MNE C (https://martinos.org/mne/stable/install_mne_c.html) (this has command line functions, which, espeically "mne_analyze" have useful GUIs, in that case for coordinate frame coregistration (MRI to MEEG).
5. (Optional) Add "alias mb 'cd /Path/to/MEEGbuddyDemo/or/your/project; source /Path/to/where/you/want/to/store/your/venv/bin/activate; export FREESURFER_HOME=/Path/to/freesurfer; source $FREESURFER_HOME/SetUpFreeSurfer.sh' " so that you can type "mb" into a terminal to activate your project. Note you need to have installed freesurfer for the last two commands in the alias (export ... and source $FREESURFER...)

To run the demo:
1. install jupyter if you haven't already: run "pip install jupyter"
2. install dependencies if you haven't already: run "pip install -r requirements.txt" OR run "pip install pandas mne scipy autoreject seaborn matplotlib tqdm joblib nitime pysurfer naturalneighbor mayavi"
3. run "jupyter notebook"

This was originally designed in python2 but now is only supported in python3

## Citation 
AP. Rockhill (2018). MEEGbuddy. GitHub Repository. https://github.com/alexrockhill/MEEGbuddy.

## Licensing
This software is OSI Certified Open Source Software.
OSI Certified is a certification mark of the Open Source Initiative.

Copyright (c) 2018, authors of MEEGbuddy
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the names of MEEGbuddy authors nor the names of any
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

**This software is provided by the copyright holders and contributors
"as is" and any express or implied warranties, including, but not
limited to, the implied warranties of merchantability and fitness for
a particular purpose are disclaimed. In no event shall the copyright
owner or contributors be liable for any direct, indirect, incidental,
special, exemplary, or consequential damages (including, but not
limited to, procurement of substitute goods or services; loss of use,
data, or profits; or business interruption) however caused and on any
theory of liability, whether in contract, strict liability, or tort
(including negligence or otherwise) arising in any way out of the use
of this software, even if advised of the possibility of such
damage.**
