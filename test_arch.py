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
parser.add_argument('--model', choices=['resnet_18', 'resnet_34', 'resnet_50', 'TDNN', 'all'], default='all', help='Model arch according to input type')
parser.add_argument('--latent-size', type=int, default=200, metavar='S', help='latent layer dimension (default: 200)')
parser.add_argument('--ncoef', type=int, default=100, metavar='N', help='number of MFCCs (default: 100)')
parser.add_argument('--pase-cfg', type=str, metavar='Path', help='Path to pase cfg')
parser.add_argument('--pase-cp', type=str, default=None, metavar='Path', help='Path to pase cp')
args = parser.parse_args()

if args.model == 'resnet_18' or args.model == 'all':
	batch = torch.rand(3, 10000)
	model = model_.ResNet_18(pase_cfg=args.pase_cfg, pase_cp=args.pase_cp, n_z=args.latent_size, ncoef=args.ncoef, proj_size=10, sm_type='softmax')
	mu = model.forward(batch)
	out = model.out_proj(mu, torch.ones(mu.size(0)))
	print('resnet_18', mu.size(), out.size())
if args.model == 'resnet_34' or args.model == 'all':
	batch = torch.rand(3, 10000)
	model = model_.ResNet_34(pase_cfg=args.pase_cfg, pase_cp=args.pase_cp, n_z=args.latent_size, ncoef=args.ncoef, proj_size=10, sm_type='softmax')
	mu = model.forward(batch)
	out = model.out_proj(mu, torch.ones(mu.size(0)))
	print('resnet_34', mu.size(), out.size())
if args.model == 'resnet_50' or args.model == 'all':
	batch = torch.rand(3, 10000)
	model = model_.ResNet_50(pase_cfg=args.pase_cfg, pase_cp=args.pase_cp, n_z=args.latent_size, ncoef=args.ncoef, proj_size=10, sm_type='softmax')
	mu = model.forward(batch)
	out = model.out_proj(mu, torch.ones(mu.size(0)))
	print('resnet_50', mu.size(), out.size())
if args.model == 'TDNN' or args.model == 'all':
	batch = torch.rand(3, 10000)
	model = model_.TDNN(pase_cfg=args.pase_cfg, pase_cp=args.pase_cp, n_z=args.latent_size, ncoef=args.ncoef, proj_size=10, sm_type='softmax')
	mu = model.forward(batch)
	out = model.out_proj(mu, torch.ones(mu.size(0)))
	print('TDNN', mu.size(), out.size())
