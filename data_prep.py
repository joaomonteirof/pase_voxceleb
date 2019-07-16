import argparse
import h5py
import numpy as np
import os
import librosa
from utils.utils import *

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Data preparation and storage in .hdf')
	parser.add_argument('--data-info-path', type=str, default='./data/', metavar='Path', help='Path to spk2utt and utt2spk')
	parser.add_argument('--out-path', type=str, default='./', metavar='Path', help='Path to output hdf file')
	parser.add_argument('--out-name', type=str, default='train.hdf', metavar='Path', help='Output hdf file name')
	parser.add_argument('--sampling-rate', type=int, default=16000, help='Sampling rate (Default: 16000)')
	parser.add_argument('--m4a', action='store_true', default=False)
	args = parser.parse_args()

	if os.path.isfile(args.out_path+args.out_name):
		os.remove(args.out_path+args.out_name)
		print(args.out_path+args.out_name+' Removed')

	utt2spk = read_utt2spk(args.data_info_path+'utt2spk')
	spk2utt = read_spk2utt(args.data_info_path+'spk2utt', 1)
	utt2rec = read_utt2rec(args.data_info_path+'wav.scp', args.m4a)

	speakers_list = list(spk2utt.keys())

	print('Start of data preparation')

	hdf = h5py.File(args.out_path+args.out_name, 'a')

	for spk in speakers_list:
		hdf.create_group(spk)

	for utt_id, rec in utt2rec.items():

		print('Processing file {}'.format(file_))

		data_, fs = librosa.load(rec, sr=args.sampling_rate)

		if not utt_id in utt2spk:
			continue

		speaker = utt2spk[utt_id]

		if not speaker in spk2utt:
			continue

		if data_.shape[0]>0:
			features = np.expand_dims(features, 0)
			hdf[speaker].create_dataset(utt_id, data=data_, maxshape=(features.shape[0]))
		else:
			print('EMPTY FEATURES ARRAY IN FILE {} !!!!!!!!!'.format(utt_id))

	hdf.close()
