from __future__ import print_function
import argparse
import torch
from train_loop import TrainLoop
import torch.optim as optim
import torch.utils.data
import model as model_
import numpy as np
from data_load import Loader, Loader_valid
import os
import sys
from utils.utils import *

# Training settings
parser = argparse.ArgumentParser(description='Speaker recognition')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--valid-batch-size', type=int, default=64, metavar='N', help='input batch size for valid (default: 64)')
parser.add_argument('--epochs', type=int, default=500, metavar='N', help='number of epochs to train (default: 500)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.001)')
parser.add_argument('--lr-pase', type=float, default=None, metavar='LR', help='Pase LR')
parser.add_argument('--momentum', type=float, default=0.9, metavar='m', help='Momentum paprameter (default: 0.9)')
parser.add_argument('--l2', type=float, default=1e-5, metavar='L2', help='Weight decay coefficient (default: 0.00001)')
parser.add_argument('--patience', type=int, default=10, metavar='S', help='Epochs to wait before decreasing LR by a factor of 0.5 (default: 10)')
parser.add_argument('--checkpoint-epoch', type=int, default=None, metavar='N', help='epoch to load for checkpointing. If None, training starts from scratch')
parser.add_argument('--checkpoint-path', type=str, default=None, metavar='Path', help='Path for checkpointing')
parser.add_argument('--pretrained-path', type=str, default=None, metavar='Path', help='Path for pre trained model')
parser.add_argument('--pase-cfg', type=str, metavar='Path', help='Path to pase cfg')
parser.add_argument('--pase-cp', type=str, default=None, metavar='Path', help='Path to pase cp')
parser.add_argument('--train-hdf-file', type=str, default='./data/train.hdf', metavar='Path', help='Path to hdf data')
parser.add_argument('--valid-hdf-file', type=str, default=None, metavar='Path', help='Path to hdf data')
parser.add_argument('--model', choices=['resnet_18', 'resnet_34', 'resnet_50', 'TDNN', 'TDNN_mfcc', 'MLP', 'global_MLP', 'pyr_rnn'], default='resnet_18', help='Model arch according to input type')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--save-every', type=int, default=1, metavar='N', help='how many epochs to wait before logging training status. Default is 1')
parser.add_argument('--ncoef', type=int, default=100, metavar='N', help='number of features output by encoder (default: 100)')
parser.add_argument('--latent-size', type=int, default=200, metavar='S', help='latent layer dimension (default: 200)')
parser.add_argument('--max-len', type=int, default=96000, metavar='N', help='maximum length per utterance (default: 96000)')
parser.add_argument('--softmax', choices=['softmax', 'am_softmax'], default='none', help='Softmax type')
parser.add_argument('--vad', action='store_true', default=True, help='Remove silence frames from train recordings')
parser.add_argument('--mfcc', action='store_true', default=False, help='MFCC')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
parser.add_argument('--no-cp', action='store_true', default=False, help='Disables checkpointing')
parser.add_argument('--train-pase', action='store_true', default=False, help='Enables PASE updates')
parser.add_argument('--verbose', type=int, default=1, metavar='N', help='Verbose is activated if > 0')
args = parser.parse_args()
args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

torch.manual_seed(args.seed)
if args.cuda:
	torch.cuda.manual_seed(args.seed)

train_dataset = Loader(hdf5_name = args.train_hdf_file, max_len = args.max_len, vad = args.vad, ncoef = args.ncoef if args.mfcc else None)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, worker_init_fn=set_np_randomseed)

if args.valid_hdf_file is not None:
	valid_dataset = Loader_valid(hdf5_name = args.valid_hdf_file, max_len = args.max_len, vad = args.vad, ncoef = args.ncoef if args.mfcc else None)
	valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.valid_batch_size, shuffle=True, num_workers=args.workers, worker_init_fn=set_np_randomseed)
else:
	valid_loader=None

if args.cuda:
	device = get_freer_gpu()
	import cupy
	cupy.cuda.Device(int(str(device).split(':')[-1])).use()
else:
	device = None

if args.model == 'resnet_18':
	model = model_.ResNet_18(pase_cfg=args.pase_cfg, pase_cp=args.pase_cp, n_z=args.latent_size, proj_size=train_dataset.n_speakers, ncoef=args.ncoef, sm_type=args.softmax)
elif args.model == 'resnet_34':
	model = model_.ResNet_34(pase_cfg=args.pase_cfg, pase_cp=args.pase_cp, n_z=args.latent_size, proj_size=train_dataset.n_speakers, ncoef=args.ncoef, sm_type=args.softmax)
elif args.model == 'resnet_50':
	model = model_.ResNet_50(pase_cfg=args.pase_cfg, pase_cp=args.pase_cp, n_z=args.latent_size, proj_size=train_dataset.n_speakers, ncoef=args.ncoef, sm_type=args.softmax)
elif args.model == 'TDNN':
	model = model_.TDNN(pase_cfg=args.pase_cfg, pase_cp=args.pase_cp, n_z=args.latent_size, proj_size=train_dataset.n_speakers, ncoef=args.ncoef, sm_type=args.softmax)
elif args.model == 'TDNN_mfcc':
	model = model_.TDNN_mfcc(n_z=args.latent_size, proj_size=train_dataset.n_speakers, ncoef=args.ncoef, sm_type=args.softmax)
elif args.model == 'MLP':
	model = model_.MLP(pase_cfg=args.pase_cfg, pase_cp=args.pase_cp, n_z=args.latent_size, proj_size=train_dataset.n_speakers, ncoef=args.ncoef, sm_type=args.softmax)
elif args.model == 'global_MLP':
	model = model_.global_MLP(pase_cfg=args.pase_cfg, pase_cp=args.pase_cp, n_z=args.latent_size, proj_size=train_dataset.n_speakers, ncoef=args.ncoef, sm_type=args.softmax)
elif args.model == 'pyr_rnn':
	model = model_.pyr_rnn(pase_cfg=args.pase_cfg, pase_cp=args.pase_cp, n_z=args.latent_size, proj_size=train_dataset.n_speakers, ncoef=args.ncoef, sm_type=args.softmax)

if args.pretrained_path is not None:
	ckpt = torch.load(args.pretrained_path, map_location = lambda storage, loc: storage)

	try:
		model.load_state_dict(ckpt['model_state'], strict=True)
	except RuntimeError as err:
		print("Runtime Error: {0}".format(err))
	except:
		print("Unexpected error:", sys.exc_info()[0])
		raise

if args.cuda:
	model = model.to(device)

optimizer_pase = optim.SGD(model.encoder.parameters(), lr=args.lr if args.lr_pase else args.lr/10., momentum=0.1, weight_decay=1e-5) if args.train_pase else None
optimizer = optim.SGD(list(model.model.parameters())+list(model.pooling.parameters())+list(model.post_pooling.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=args.l2) if not args.model=='global_MLP' else optim.SGD(model.model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.l2)

trainer = TrainLoop(model, optimizer, optimizer_pase, train_loader, valid_loader, patience=args.patience, verbose=args.verbose, device=device, save_cp=(not args.no_cp), checkpoint_path=args.checkpoint_path, checkpoint_epoch=args.checkpoint_epoch, cuda=args.cuda)

if args.verbose >0:
	print(' ')
	print('Cuda Mode: {}'.format(args.cuda))
	print('Device: {}'.format(device))
	print('Train PASE: {}'.format(args.train_pase))
	print('Softmax Mode: {}'.format(args.softmax))
	print('Selected model: {}'.format(args.model))
	print('Embeddings size: {}'.format(args.latent_size))
	print('Batch size: {}'.format(args.batch_size))
	print('LR: {}'.format(args.lr))
	print('Pase LR: {}'.format(args.lr_pase if args.lr_pase else args.lr/10.))
	print('momentum: {}'.format(args.momentum))
	print('l2: {}'.format(args.l2))
	print('Patience: {}'.format(args.patience))
	print('Max input length: {}'.format(args.max_len))
	print('Number of train speakers: {}'.format(train_dataset.n_speakers))
	print('Number of train examples: {}'.format(len(train_dataset.utt_list)))
	print('Number of valid speakers: {}'.format(valid_dataset.n_speakers))
	print('Number of valid examples: {}'.format(len(valid_dataset.utt_list)))
	print(' ')

trainer.train(n_epochs=args.epochs, save_every=args.save_every)
