import argparse
import h5py
import numpy as np
import glob
import os
import librosa

def read_utt2spk(path):
	with open(path, 'r') as file:
		pairs = file.readlines()

	utt2spk_dict = {}

	for pair in pairs:
		utt, spk = pair.split(' ')
		utt2spk_dict[utt] = spk.replace('\n','')

	return utt2spk_dict

def read_spk2utt(path, min_recs):
	with open(path, 'r') as file:
		rows = file.readlines()

	spk2utt_dict = {}

	for row in rows:
		spk_utts = row.replace('\n','').split(' ')
		if len(spk_utts[1:])>=min_recs:
			spk2utt_dict[spk_utts[0]] = spk_utts[1:]

	return spk2utt_dict

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Data preparation and storage in .hdf')
	parser.add_argument('--path-to-data', type=str, default='./data/', metavar='Path', help='Path to wav files')
	parser.add_argument('--data-info-path', type=str, default='./data/', metavar='Path', help='Path to spk2utt and utt2spk')
	parser.add_argument('--spk2utt', type=str, default=None, metavar='Path', help='Path to spk2utt')
	parser.add_argument('--utt2spk', type=str, default=None, metavar='Path', help='Path to utt2spk')
	parser.add_argument('--path-to-more-data', type=str, default=None, metavar='Path', help='Path to extra data')
	parser.add_argument('--more-data-info-path', type=str, default=None, metavar='Path', help='Path to spk2utt and utt2spk')
	parser.add_argument('--more-spk2utt', type=str, default=None, metavar='Path', help='Path to spk2utt')
	parser.add_argument('--more-utt2spk', type=str, default=None, metavar='Path', help='Path to utt2spk')
	parser.add_argument('--out-path', type=str, default='./', metavar='Path', help='Path to output hdf file')
	parser.add_argument('--out-name', type=str, default='train.hdf', metavar='Path', help='Output hdf file name')
	parser.add_argument('--min-recordings', type=int, default=-1, help='Minimum number of recordings per speaker')
	parser.add_argument('--sampling-rate', type=int, default=16000, help='Sampling rate (Default: 16000)')
	args = parser.parse_args()

	if os.path.isfile(args.out_path+args.out_name):
		os.remove(args.out_path+args.out_name)
		print(args.out_path+args.out_name+' Removed')

	utt2spk = read_utt2spk(args.utt2spk if args.utt2spk else args.data_info_path+'utt2spk')
	spk2utt = read_spk2utt(args.spk2utt if args.spk2utt else args.data_info_path+'spk2utt', args.min_recordings)

	wav_list = glob.glob(args.path_to_data + '*.wav')

	if args.path_to_more_data:
		wav_list.extend(glob.glob(args.path_to_more_data + '*.wav'))
		utt2spk = {**utt2spk, **read_utt2spk(args.more_utt2spk if args.more_utt2spk else args.more_data_info_path+'utt2spk')}
		spk2utt = {**spk2utt, **read_spk2utt(args.spk2utt if args.more_spk2utt else args.more_data_info_path+'spk2utt', args.min_recordings)}

	if len(scp_list)<1:
		print('Nothing found at {}.'.format(args.path_to_data))
		exit(1)

	speakers_list = list(spk2utt.keys())

	print('Start of data preparation')

	hdf = h5py.File(args.out_path+args.out_name, 'a')

	for spk in speakers_list:
		hdf.create_group(spk)

	for file_ in wav_list:

		print('Processing file {}'.format(file_))

		data_, fs = librosa.load(file_, sr=args.sampling_rate)

		utt_id = data_.split('/')[-1].split('.wav')[0]

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
