# MEEGbuddy

I work in a lab that analyzes many psychphysical tasks with the same structure: there are trials with stimuli and responses. If this matches the description of what you're studying, this tool will probably be very helpful. Using MEEGbuddy, I was able to preprocess ~70 subjects for three tasks using both MEG and EEG in three days of manual input.

MEEGbuddy uses MNE, autoreject and PCI (noreun on github) for preprocessing mostly but also analysis of MEEG data. TFR psd multitapers, including sleep scoring, and morlet are supported as well as cluster permutation tests and lempel-ziv complexity (connectivity coming soon hopefully). These preprocessing and analysis steps all come with defaults that will run well as is and have lots of the kinks figured out already with the added bonus that they are saved in BIDS structure (or pretty close).

## Installation Instructions
1. clone into a repository
2. run python setup.py install

To run the demo:
1. install jupyter if you haven't already: pip install jupyter
2. install dependencies: pip install pandas mne scipy autoreject seaborn matplotlib tqdm joblib nitime pysurfer naturalneighbor mayavi
3. run "jupyter-notebook" from a terminal

This was originally designed in python2 but now is only supported in python3, it will probably work in python2 for a while longer but no promises

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
