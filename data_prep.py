import argparse
import h5py
import numpy as np
import os
import librosa
from utils.utils import *

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Data preparation and storage in .hdf')
	parser.add_argument('--data-info-path', type=str, default=None, metavar='Path', help='Path to wav.scp, spk2utt and utt2spk')
	parser.add_argument('--data-path', type=str, default=None, metavar='Path', help='Path to wav.scp, spk2utt and utt2spk')
	parser.add_argument('--out-path', type=str, default='./', metavar='Path', help='Path to output hdf file')
	parser.add_argument('--out-name', type=str, default='train.hdf', metavar='Path', help='Output hdf file name')
	parser.add_argument('--sampling-rate', type=int, default=16000, help='Sampling rate (Default: 16000)')
	parser.add_argument('--m4a', action='store_true', default=False)
	args = parser.parse_args()

	if os.path.isfile(args.out_path+args.out_name):
		os.remove(args.out_path+args.out_name)
		print(args.out_path+args.out_name+' Removed')

	if args.data_path is not None:
		import glob
		file_list=glob.glob(args.data_path+'/**/*.wav', recursive=True)
		utt2spk = {}
		spk2utt = {}
		utt2rec = {}
		for file_ in file_list:
			file_split = file_.split('/')
			utt = file_split[-3]+'_'+file_split[-2]+'_'+file_split[-1].split('.')[0]
			spk = file_split[-3]
			utt2spk[utt]=spk
			utt2rec[utt]=file_
			if spk in spk2utt:
				spk2utt[spk].append(utt)
			else:
				spk2utt[spk]=[utt]

	elif args.data_info_path is not None:
		utt2spk = read_utt2spk(args.data_info_path+'utt2spk')
		spk2utt = read_spk2utt(args.data_info_path+'spk2utt', 1)
		utt2rec = read_utt2rec(args.data_info_path+'wav.scp', args.m4a)
	else:
		print('\nSet either --data-path or --data-info-path\n')
		exit(1)

	speakers_list = list(spk2utt.keys())

	print('Start of data preparation')

	hdf = h5py.File(args.out_path+args.out_name, 'a')

	for spk in speakers_list:
		hdf.create_group(spk)

	for utt_id, rec in utt2rec.items():

		data_, fs = librosa.load(rec, sr=args.sampling_rate)

		if not utt_id in utt2spk:
			continue

		speaker = utt2spk[utt_id]

		if not speaker in spk2utt:
			continue

		if data_.shape[0]>0:
			hdf[speaker].create_dataset(utt_id, data=data_)
		else:
			print('EMPTY FEATURES ARRAY IN FILE {} !!!!!!!!!'.format(utt_id))

	hdf.close()
