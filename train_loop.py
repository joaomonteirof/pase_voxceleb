import torch
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm

import os

from utils.harvester import AllTripletSelector
from utils.losses import LabelSmoothingLoss
from utils.utils import compute_eer

class TrainLoop(object):

	def __init__(self, model, optimizer, optimizer_pase, train_loader, valid_loader, label_smoothing, patience, verbose=-1, device=0, cp_name=None, save_cp=False, checkpoint_path=None, checkpoint_epoch=None, cuda=True):
		if checkpoint_path is None:
			# Save to current directory
			self.checkpoint_path = os.getcwd()
		else:
			self.checkpoint_path = checkpoint_path
			if not os.path.isdir(self.checkpoint_path):
				os.mkdir(self.checkpoint_path)

		self.save_epoch_fmt = os.path.join(self.checkpoint_path, cp_name) if cp_name else os.path.join(self.checkpoint_path, 'checkpoint_{}ep.pt')
		self.cuda_mode = cuda
		self.model = model
		self.optimizer = optimizer
		self.optimizer_pase = optimizer_pase
		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.total_iters = 0
		self.cur_epoch = 0
		self.harvester_all = AllTripletSelector()
		self.verbose = verbose
		self.save_cp = save_cp
		self.device = device
		self.history = {'train_loss': [], 'train_loss_batch': []}

		if label_smoothing>0.0:
			self.ce_criterion = LabelSmoothingLoss(label_smoothing, lbl_set_size=train_loader.dataset.n_speakers)
		else:
			self.ce_criterion = torch.nn.CrossEntropyLoss()

		if self.valid_loader is not None:
			self.history['valid_loss'] = []
			self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5, patience=patience, verbose=True if self.verbose>0 else False, threshold=1e-4, min_lr=1e-7)
		else:
			self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[20, 100, 200, 300, 400], gamma=0.1)

		if checkpoint_epoch is not None:
			self.load_checkpoint(self.save_epoch_fmt.format(checkpoint_epoch))

	def train(self, n_epochs=1, save_every=1):

		while (self.cur_epoch < n_epochs):

			np.random.seed()
			self.train_loader.dataset.update_lists()

			if self.verbose>0:
				print(' ')
				print('Epoch {}/{}'.format(self.cur_epoch+1, n_epochs))
				print('Number of training examples given new list: {}'.format(len(self.train_loader.dataset)))
				train_iter = tqdm(enumerate(self.train_loader))
			else:
				train_iter = enumerate(self.train_loader)

			train_loss_epoch=0.0

			for t, batch in train_iter:
				train_loss = self.train_step(batch)
				self.history['train_loss_batch'].append(train_loss)
				train_loss_epoch+=train_loss
				self.total_iters += 1

			self.history['train_loss'].append(train_loss_epoch/(t+1))

			if self.verbose>0:
				print('Total train loss, {:0.4f}'.format(self.history['train_loss'][-1]))

			if self.valid_loader is not None:

				scores, labels = None, None

				for t, batch in enumerate(self.valid_loader):
					scores_batch, labels_batch = self.valid(batch)

					try:
						scores = np.concatenate([scores, scores_batch], 0)
						labels = np.concatenate([labels, labels_batch], 0)
					except:
						scores, labels = scores_batch, labels_batch

				self.history['valid_loss'].append(compute_eer(labels, scores))
				if self.verbose>0:
					print('Current validation loss, best validation loss, and epoch: {:0.4f}, {:0.4f}, {}'.format(self.history['valid_loss'][-1], np.min(self.history['valid_loss']), 1+np.argmin(self.history['valid_loss'])))

				self.scheduler.step(self.history['valid_loss'][-1])

			else:
				self.scheduler.step()

			if self.verbose>0:
				print('Current LR: {}'.format(self.optimizer.param_groups[0]['lr']))
				if self.optimizer_pase:
					print('Current LR PASE: {}'.format(self.optimizer_pase.param_groups[0]['lr']))

			self.cur_epoch += 1

			if self.valid_loader is not None and self.save_cp and (self.cur_epoch % save_every == 0 or self.history['valid_loss'][-1] < np.min([np.inf]+self.history['valid_loss'][:-1])):
					self.checkpointing()
			elif self.save_cp and self.cur_epoch % save_every == 0:
					self.checkpointing()

		if self.verbose>0:
			print('Training done!')

		if self.valid_loader is not None:
			if self.verbose>0:
				print('Best validation loss and corresponding epoch: {:0.4f}, {}'.format(np.min(self.history['valid_loss']), 1+np.argmin(self.history['valid_loss'])))

			return np.min(self.history['valid_loss'])
		else:
			return np.min(self.history['train_loss'])

	def train_step(self, batch):

		self.model.train()
		self.optimizer.zero_grad()
		if self.optimizer_pase:
			self.optimizer_pase.zero_grad()
		else:
			try:
				self.model.encoder.eval()
			except AttributeError:
				pass

		utterances, y = batch
		y = y.squeeze()

		ridx = np.random.randint(utterances.size(-1)//2, utterances.size(-1))

		try:
			utterances = utterances[:,:ridx]
		except:
			utterances = utterances[:,:,:ridx]

		if self.cuda_mode:
			utterances = utterances.to(self.device)
			y = y.to(self.device)

		embeddings = self.model.forward(utterances)
		embeddings_norm = F.normalize(embeddings, p=2, dim=1)

		loss = self.ce_criterion(self.model.out_proj(embeddings_norm, y), y)
		loss.backward()
		self.optimizer.step()
		if self.optimizer_pase:
			self.optimizer_pase.step()
		return loss.item()

	def valid(self, batch):

		self.model.eval()

		with torch.no_grad():

			utt_1, utt_2, utt_3, utt_4, utt_5, y = batch
			utterances = torch.cat([utt_1, utt_2, utt_3, utt_4, utt_5], dim=0)
			y = torch.cat(5*[y], dim=0).squeeze()

			ridx = np.random.randint(utterances.size(-1)//2, utterances.size(-1))

			try:
				utterances = utterances[:,:ridx]
			except:
				utterances = utterances[:,:,:ridx]

			if self.cuda_mode:
				utterances = utterances.to(self.device)
				y = y.to(self.device)

			embeddings = self.model.forward(utterances)
			embeddings_norm = F.normalize(embeddings, p=2, dim=1)

			triplets_idx = self.harvester_all.get_triplets(embeddings_norm.detach(), y)

			if self.cuda_mode:
				triplets_idx = triplets_idx.cuda(self.device)

			emb_a = torch.index_select(embeddings_norm, 0, triplets_idx[:, 0])
			emb_p = torch.index_select(embeddings_norm, 0, triplets_idx[:, 1])
			emb_n = torch.index_select(embeddings_norm, 0, triplets_idx[:, 2])

			scores_p = F.cosine_similarity(emb_a, emb_p)
			scores_n = F.cosine_similarity(emb_a, emb_n)

		return np.concatenate([scores_p.detach().cpu().numpy(), scores_n.detach().cpu().numpy()], 0), np.concatenate([np.ones(scores_p.size(0)), np.zeros(scores_n.size(0))], 0)

	def checkpointing(self):

		# Checkpointing
		if self.verbose>0:
			print('Checkpointing...')
		if self.optimizer_pase:
			ckpt = {'model_state': self.model.state_dict(),
			'optimizer_state': self.optimizer.state_dict(),
			'optimizer_pase_state': self.optimizer_pase.state_dict(),
			'scheduler_state': self.scheduler.state_dict(),
			'history': self.history,
			'total_iters': self.total_iters,
			'cur_epoch': self.cur_epoch}
		else:
			ckpt = {'model_state': self.model.state_dict(),
			'optimizer_state': self.optimizer.state_dict(),
			'scheduler_state': self.scheduler.state_dict(),
			'history': self.history,
			'total_iters': self.total_iters,
			'cur_epoch': self.cur_epoch}
		try:
			torch.save(ckpt, self.save_epoch_fmt.format(self.cur_epoch))
		except:
			torch.save(ckpt, self.save_epoch_fmt)

	def load_checkpoint(self, ckpt):

		if os.path.isfile(ckpt):

			ckpt = torch.load(ckpt, map_location = lambda storage, loc: storage)
			# Load model state
			self.model.load_state_dict(ckpt['model_state'])
			# Load optimizer state
			self.optimizer.load_state_dict(ckpt['optimizer_state'])
			if self.optimizer_pase:
				self.optimizer_pase.load_state_dict(ckpt['optimizer_pase_state'])
			# Load scheduler state
			self.scheduler.load_state_dict(ckpt['scheduler_state'])
			# Load history
			self.history = ckpt['history']
			self.total_iters = ckpt['total_iters']
			self.cur_epoch = ckpt['cur_epoch']
			if self.cuda_mode:
				self.model = self.model.cuda(self.device)

		else:
			print('No checkpoint found at: {}'.format(ckpt))

	def print_grad_norms(self):
		norm = 0.0
		for params in list(self.model.parameters()):
			norm+=params.grad.norm(2).item()
		print('Sum of grads norms: {}'.format(norm))
