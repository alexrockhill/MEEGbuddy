# Conversion for .easy (with accompanying .info) file to .fif file (EEGLAB format to MNE format)
# 5.24.2019 Alex Rockhill aprockhill206@gmail.com

import sys, os
import os.path as op
import numpy as np
from pandas import read_csv, DataFrame
from mne.io import RawArray
from mne import create_info, Annotations
from mne.channels import read_montage

channel_conversion = {'Ch1': 'Fp1', 'Ch2': 'Fp2', 'Ch3': 'F3', 'Ch4': 'F4', 
     				  'Ch5': 'Fz', 'Ch6': 'P7', 'Ch7': 'P8', 'Ch8': 'Oz'}

montage = read_montage('standard_1005', transform=False)
montage_kind = 'Clincal 8 Channel'

def easy2fif(easyf):
	infof = op.join(op.dirname(easyf), op.basename(easyf.replace('.easy', '.info')))
	info_dict = dict()
	char_counter = 0
	with open(infof, 'r') as f:
		for line in f:
			char_counter += len(line)
			line = line.rstrip()
			if line:
				try:
					colon_index = line.index(':')
				except Exception as e:
					print(e, 'Unexpected info file format for line %s' % line)
				if colon_index == len(line) - 1:
					key = line[:colon_index]
					val = ''
					char_counter2 = 0
					while True:
						line = f.readline()
						char_counter2 += len(line)
						line = line.rstrip()
						if not line:
							break
						val += line
					char_counter += char_counter2
					f.seek(char_counter)
					info_dict[key] = val
				else:
					key, val = line[:colon_index], line[colon_index+2:]
					info_dict[key] = val
	with open(easyf, 'r') as f:
		n_times = 0
		for line in f:
			n_times += 1
		n_channels = len(line.split('\t')) - 1  # exclude time stamp channel
		if n_channels != int(info_dict['Number of EEG channels']) + 1: # event channel
			raise ValueError('Easy and info file disagreement on number of channels')
		f.seek(0)
		arr = np.zeros((n_channels, n_times))
		for j, line in enumerate(f):
			for i, val in enumerate(line.split('\t')[:n_channels]):
				arr[i, j] = val
	if info_dict['EEG units'] == 'nV':
		arr[:n_channels-1] *= 1e-9
	elif info_dict['EEG units'] == 'μV':
		arr[:n_channels-1] *= 1e-6
	elif info_dict['EEG units'] == 'mV':
		arr[:n_channels-1] *= 1e-3
	else:
		raise ValueError('Unrecognized EEG unit type %s' % info_dict['EEG units'])

	ch_names2 = [l.split(':')[-1].strip() for l in info_dict['EEG montage'].strip().split('\t')] + ['Events']
	ch_names = [channel_conversion[ch] if ch in channel_conversion else ch for ch in ch_names2]
	ch_types = ['eeg' for i in range(int(info_dict['Number of EEG channels']))] + ['stim']
	sfreq = float(info_dict['EEG sampling rate'].split(' ')[0])
	montage.kind = montage_kind
	exclude_indices = [i for i, ch in enumerate(montage.ch_names) if ch not in (ch_names + ['LPA', 'RPA', 'Nz'])]
	montage.pos = np.delete(montage.pos, exclude_indices, axis=0)
	montage.ch_names = [ch for ch in montage.ch_names if ch in (ch_names + ['LPA', 'RPA', 'Nz'])]
	montage.selection = np.arange(len(montage.ch_names))
	info = create_info(ch_names, sfreq, ch_types, montage=montage, verbose=False)
	raw = RawArray(arr, info, verbose=False)
	return raw
	#raw.save(op.join(op.dirname(easyf), op.basename(easyf.replace('.easy', '-raw.fif'))))
	#raw.set_eeg_reference(ref_channels='average', projection=False)

	'''test
	from mne import find_events, Epochs
	events = find_events(raw, event_ch='Events')
	epochs = Epochs(raw, events)
	for this_id in epochs.event_id:
		epochs[this_id].average().plot(window_title=this_id)'''


if __name__ == '__main__':
	if len(sys.argv) != 2:
		raise ValueError('Easy file name required')
	easy2fif(sys.argv[1])