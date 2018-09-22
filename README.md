# MEEGbuddy

I work in a lab that analyzes many psychphysical tasks with the same structure: there are trials with stimuli and responses. If this matches the description of what you're studying, this tool will probably be very helpful. Using MEEGbuddy, I was able to preprocess ~70 subjects for three tasks using both MEG and EEG in three days of manual input.

MEEGbuddy uses MNE, autoreject and PCI (noreun on github) for preprocessing mostly but also analysis of MEEG data. TFR psd multitapers, including sleep scoring, and morlet are supported as well as cluster permutation tests and lempel-ziv complexity (connectivity coming soon hopefully). These preprocessing and analysis steps all come with defaults that will run well as is and have lots of the kinks figured out already with the added bonus that they are saved in BIDS structure (or pretty close).

## Installation Instructions
clone into a repository run python setup.py install
This was originally designed in python2 but now is only supported in python3, it will probably work in python2 for a while longer but no promises
