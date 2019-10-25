from __future__ import print_function
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data
import model as model_
from utils.utils import *

# Training settings
parser = argparse.ArgumentParser(description='Test new architectures')
parser.add_argument('--model', choices=['resnet_18', 'resnet_34', 'resnet_50', 'TDNN', 'TDNN_mfcc', 'MLP', 'global_MLP', 'pyr_rnn', 'all'], default='all', help='Model arch according to input type')
parser.add_argument('--latent-size', type=int, default=200, metavar='S', help='latent layer dimension (default: 200)')
parser.add_argument('--ncoef', type=int, default=100, metavar='N', help='number of MFCCs (default: 100)')
parser.add_argument('--pase-cfg', type=str, metavar='Path', help='Path to pase cfg')
parser.add_argument('--pase-cp', type=str, default=None, metavar='Path', help='Path to pase cp')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
args = parser.parse_args()
args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

if args.cuda:
	device = get_freer_gpu()
	import cupy
	cupy.cuda.Device(int(str(device).split(':')[-1])).use()
else:
	device = torch.device('cpu')

if args.model == 'resnet_18' or args.model == 'all':
	batch = torch.rand(3, 10000).to(device)
	model = model_.ResNet_18(pase_cfg=args.pase_cfg, pase_cp=args.pase_cp, n_z=args.latent_size, ncoef=args.ncoef, proj_size=10, sm_type='softmax').to(device)
	mu = model.forward(batch)
	out = model.out_proj(mu, torch.ones(mu.size(0)))
	print('resnet_18', mu.size(), out.size())
if args.model == 'resnet_34' or args.model == 'all':
	batch = torch.rand(3, 10000).to(device)
	model = model_.ResNet_34(pase_cfg=args.pase_cfg, pase_cp=args.pase_cp, n_z=args.latent_size, ncoef=args.ncoef, proj_size=10, sm_type='softmax').to(device)
	mu = model.forward(batch)
	out = model.out_proj(mu, torch.ones(mu.size(0)))
	print('resnet_34', mu.size(), out.size())
if args.model == 'resnet_50' or args.model == 'all':
	batch = torch.rand(3, 10000).to(device)
	model = model_.ResNet_50(pase_cfg=args.pase_cfg, pase_cp=args.pase_cp, n_z=args.latent_size, ncoef=args.ncoef, proj_size=10, sm_type='softmax').to(device)
	mu = model.forward(batch)
	out = model.out_proj(mu, torch.ones(mu.size(0)))
	print('resnet_50', mu.size(), out.size())
if args.model == 'TDNN' or args.model == 'all':
	batch = torch.rand(3, 10000).to(device)
	model = model_.TDNN(pase_cfg=args.pase_cfg, pase_cp=args.pase_cp, n_z=args.latent_size, ncoef=args.ncoef, proj_size=10, sm_type='softmax').to(device)
	mu = model.forward(batch)
	out = model.out_proj(mu, torch.ones(mu.size(0)))
	print('TDNN', mu.size(), out.size())
if args.model == 'TDNN_mfcc' or args.model == 'all':
	batch = torch.rand(3, args.ncoef, 200).to(device)
	model = model_.TDNN_mfcc(n_z=args.latent_size, ncoef=args.ncoef, proj_size=10, sm_type='softmax').to(device)
	mu = model.forward(batch)
	out = model.out_proj(mu, torch.ones(mu.size(0)))
	print('TDNN_mfcc', mu.size(), out.size())
if args.model == 'MLP' or args.model == 'all':
	batch = torch.rand(3, 10000).to(device)
	model = model_.MLP(pase_cfg=args.pase_cfg, pase_cp=args.pase_cp, n_z=args.latent_size, ncoef=args.ncoef, proj_size=10, sm_type='softmax').to(device)
	mu = model.forward(batch)
	out = model.out_proj(mu, torch.ones(mu.size(0)))
	print('MLP', mu.size(), out.size())
if args.model == 'global_MLP' or args.model == 'all':
	batch = torch.rand(3, 10000).to(device)
	model = model_.global_MLP(pase_cfg=args.pase_cfg, pase_cp=args.pase_cp, n_z=args.latent_size, ncoef=args.ncoef, proj_size=10, sm_type='softmax').to(device)
	mu = model.forward(batch)
	out = model.out_proj(mu, torch.ones(mu.size(0)))
	print('global_MLP', mu.size(), out.size())
if args.model == 'pyr_rnn' or args.model == 'all':
	batch = torch.rand(3, 10000).to(device)
	model = model_.pyr_rnn(pase_cfg=args.pase_cfg, pase_cp=args.pase_cp, n_z=args.latent_size, ncoef=args.ncoef, proj_size=10, sm_type='softmax').to(device)
	mu = model.forward(batch)
	out = model.out_proj(mu, torch.ones(mu.size(0)))
	print('pyr_rnn', mu.size(), out.size())
