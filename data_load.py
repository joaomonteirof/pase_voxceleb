import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.utils import strided_app

def add_wgn(s,var=1e-4):
	np.random.seed(0)
	noise = np.random.normal(0,var,len(s))
	return s + noise

def enframe(x, win_len, hop_len):

	x = np.squeeze(x)
	if x.ndim != 1:
		raise TypeError("enframe input must be a 1-dimensional array.")
	n_frames = 1 + np.int(np.floor((len(x) - win_len) / float(hop_len)))
	x_framed = np.zeros((n_frames, win_len))
	for i in range(n_frames):
		x_framed[i] = x[i * hop_len : i * hop_len + win_len]
	return x_framed

def deframe(x_framed, win_len, hop_len):
	n_frames = len(x_framed)
	n_samples = n_frames*hop_len + win_len
	x_samples = np.zeros((n_samples,1))
	for i in range(n_frames):
		x_samples[i*hop_len : i*hop_len + win_len] = x_framed[i]
	return x_samples

def zero_mean(xframes):
	m = np.mean(xframes,axis=1)
	xframes = xframes - np.tile(m,(xframes.shape[1],1)).T
	return xframes

def compute_nrg(xframes):
	n_frames = xframes.shape[1]
	return np.diagonal(np.dot(xframes,xframes.T))/float(n_frames)

def compute_log_nrg(xframes):
	n_frames = xframes.shape[1]
	raw_nrgs = np.log(compute_nrg(xframes+1e-5))/float(n_frames)
	return (raw_nrgs - np.mean(raw_nrgs))/(np.sqrt(np.var(raw_nrgs)))

def power_spectrum(xframes):
	X = np.fft.fft(xframes,axis=1)
	X = np.abs(X[:,:X.shape[1]/2])**2
	return np.sqrt(X)

def nrg_vad(xframes,percent_thr,nrg_thr=0.,context=5):
	xframes = zero_mean(xframes)
	n_frames = xframes.shape[1]
	
	# Compute per frame energies:
	xnrgs = compute_log_nrg(xframes)
	xvad = np.zeros((n_frames,1))
	for i in range(n_frames):
		start = max(i-context,0)
		end = min(i+context,n_frames-1)
		n_above_thr = np.sum(xnrgs[start:end]>nrg_thr)
		n_total = end-start+1
		xvad[i] = 1.*((float(n_above_thr)/n_total) > percent_thr)
	return xvad

class Loader(Dataset):

	def __init__(self, hdf5_name, max_len, vad=True):
		super(Loader, self).__init__()
		self.hdf5_name = hdf5_name
		self.max_len = int(max_len)
		self.vad=vad

		self.create_lists()

		self.open_file = None

		self.update_lists()

	def __getitem__(self, index):

		utt_1, utt_2, utt_3, utt_4, utt_5, spk, y= self.utt_list[index]

		if not self.open_file: self.open_file = h5py.File(self.hdf5_name, 'r')

		utt_1_data = torch.from_numpy( self.prep_utterance( self.open_file[spk][utt_1] ) )
		utt_2_data = torch.from_numpy( self.prep_utterance( self.open_file[spk][utt_2] ) )
		utt_3_data = torch.from_numpy( self.prep_utterance( self.open_file[spk][utt_3] ) )
		utt_4_data = torch.from_numpy( self.prep_utterance( self.open_file[spk][utt_4] ) )
		utt_5_data = torch.from_numpy( self.prep_utterance( self.open_file[spk][utt_5] ) )

		return utt_1_data, utt_2_data, utt_3_data, utt_4_data, utt_5_data, y

	def __len__(self):
		return len(self.utt_list)

	def prep_utterance(self, data):

		data=data.value

		if self.vad:
			win_len = int(16000*0.025)
			hop_len = int(16000*0.010)
			sframes = enframe(data,win_len,hop_len)
			percent_high_nrg = 0.4
			vad = nrg_vad(sframes,percent_high_nrg)
			vad = deframe(vad,win_len,hop_len)[:len(data)].squeeze()
			data_ = data[np.where(vad==1)]
			if data_.shape[-1]<500:
				data_ = data
		else:
			data_ = data

		try:
			ridx = np.random.randint(0, data_.shape[-1]-self.max_len)
			data_ = data_[ridx:(ridx+self.max_len)]

		except ValueError:

			mul = int(np.ceil(self.max_len/data_.shape[-1]))
			data_ = np.tile(data_, (mul))
			data_ = data_[:self.max_len]

		return data_

	def create_lists(self):

		open_file = h5py.File(self.hdf5_name, 'r')

		self.spk2label = {}
		self.spk2utt = {}
		self.utt_list = []

		for i, spk in enumerate(open_file):
			spk_utt_list = list(open_file[spk])
			self.spk2utt[spk] = spk_utt_list
			self.spk2label[spk] = torch.LongTensor([i])

		open_file.close()

		self.n_speakers = len(self.spk2utt)

	def update_lists(self):

		self.utt_list = []

		for i, spk in enumerate(self.spk2utt):
			spk_utt_list = np.random.permutation(list(self.spk2utt[spk]))

			idxs = strided_app(np.arange(len(spk_utt_list)),5,5)

			for idxs_list in idxs:
				if len(idxs_list)==5:
					self.utt_list.append([spk_utt_list[utt_idx] for utt_idx in idxs_list])
					self.utt_list[-1].append(spk)
					self.utt_list[-1].append(self.spk2label[spk])

class Loader_valid(Dataset):

	def __init__(self, hdf5_name, max_len, vad=True):
		super(Loader_valid, self).__init__()
		self.hdf5_name = hdf5_name
		self.max_len = int(max_len)
		self.vad=vad

		self.create_lists()

		self.open_file = None

	def __getitem__(self, index):

		utt = self.utt_list[index]
		spk = self.utt2spk[utt]

		if not self.open_file: self.open_file = h5py.File(self.hdf5_name, 'r')

		utt_data = self.prep_utterance( self.open_file[spk][utt] )
		utt_data = torch.from_numpy( utt_data )

		utt_1, utt_2, utt_3, utt_4 = np.random.choice(self.spk2utt[spk], 4)

		utt_1_data = torch.from_numpy( self.prep_utterance( self.open_file[spk][utt_1] ) )
		utt_2_data = torch.from_numpy( self.prep_utterance( self.open_file[spk][utt_2] ) )
		utt_3_data = torch.from_numpy( self.prep_utterance( self.open_file[spk][utt_3] ) )
		utt_4_data = torch.from_numpy( self.prep_utterance( self.open_file[spk][utt_4] ) )

		return utt_data, utt_1_data, utt_2_data, utt_3_data, utt_4_data, self.utt2label[utt]

	def __len__(self):
		return len(self.utt_list)

	def prep_utterance(self, data):

		data=data.value

		if self.vad:
			win_len = int(16000*0.025)
			hop_len = int(16000*0.010)
			sframes = enframe(data,win_len,hop_len)
			percent_high_nrg = 0.4
			vad = nrg_vad(sframes,percent_high_nrg)
			vad = deframe(vad,win_len,hop_len)[:len(data)].squeeze()
			data_ = data[np.where(vad==1)]
			if data_.shape[-1]<500:
				data_ = data
		else:
			data_ = data

		try:
			ridx = np.random.randint(0, data_.shape[-1]-self.max_len)
			data_ = data_[ridx:(ridx+self.max_len)]

		except ValueError:

			mul = int(np.ceil(self.max_len/data_.shape[-1]))
			data_ = np.tile(data_, (mul))
			data_ = data_[:self.max_len]

		return data_

	def create_lists(self):

		open_file = h5py.File(self.hdf5_name, 'r')

		self.n_speakers = len(open_file)

		self.utt2label = {}
		self.utt2spk = {}
		self.spk2utt = {}
		self.utt_list = []

		for i, spk in enumerate(open_file):
			spk_utt_list = list(open_file[spk])
			self.spk2utt[spk] = spk_utt_list
			for utt in spk_utt_list:
				self.utt2label[utt] = torch.LongTensor([i])
				self.utt2spk[utt] = spk
				self.utt_list.append(utt)

		open_file.close()
