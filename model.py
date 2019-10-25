## SE stuff from https://github.com/moskomule/senet.pytorch/blob/master/senet/se_resnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from utils.losses import AMSoftmax, Softmax
from pase.models.frontend import wf_builder

class SelfAttention(nn.Module):
	def __init__(self, hidden_size):
		super(SelfAttention, self).__init__()

		#self.output_size = output_size
		self.hidden_size = hidden_size
		self.att_weights = nn.Parameter(torch.Tensor(1, hidden_size), requires_grad=True)

		init.kaiming_uniform_(self.att_weights)

	def forward(self, inputs):

		batch_size = inputs.size(0)
		weights = torch.bmm(inputs, self.att_weights.permute(1, 0).unsqueeze(0).repeat(batch_size, 1, 1))

		if inputs.size(0)==1:
			attentions = F.softmax(torch.tanh(weights), dim=1)
			weighted = torch.mul(inputs, attentions.expand_as(inputs))
		else:
			attentions = F.softmax(torch.tanh(weights.squeeze()),dim=1)
			weighted = torch.mul(inputs, attentions.unsqueeze(2).expand_as(inputs))

		noise = 1e-5*torch.randn(weighted.size())

		if inputs.is_cuda:
			noise = noise.cuda(inputs.get_device())

		avg_repr, std_repr = weighted.sum(1), (weighted+noise).std(1)

		representations = torch.cat((avg_repr,std_repr),1)

		return representations

class PreActBlock(nn.Module):
	'''Pre-activation version of the BasicBlock.'''
	expansion = 1

	def __init__(self, in_planes, planes, stride=1):
		super(PreActBlock, self).__init__()
		self.bn1 = nn.BatchNorm2d(in_planes)
		self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

		if stride != 1 or in_planes != self.expansion*planes:
			self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False))

	def forward(self, x):
		out = F.relu(self.bn1(x))
		shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
		out = self.conv1(out)
		out = self.conv2(F.relu(self.bn2(out)))
		out += shortcut
		return out


class PreActBottleneck(nn.Module):
	'''Pre-activation version of the original Bottleneck module.'''
	expansion = 4

	def __init__(self, in_planes, planes, stride=1):
		super(PreActBottleneck, self).__init__()
		self.bn1 = nn.BatchNorm2d(in_planes)
		self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn3 = nn.BatchNorm2d(planes)
		self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

		if stride != 1 or in_planes != self.expansion*planes:
			self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False))

	def forward(self, x):
		out = F.relu(self.bn1(x))
		shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
		out = self.conv1(out)
		out = self.conv2(F.relu(self.bn2(out)))
		out = self.conv3(F.relu(self.bn3(out)))
		out += shortcut
		return out

class ResNet_50(nn.Module):
	def __init__(self, pase_cfg, pase_cp=None, n_z=256, layers=[3,4,6,3], block=PreActBottleneck, proj_size=0, ncoef=23, sm_type='none'):
		self.in_planes = 16
		super(ResNet_50, self).__init__()

		self.model = nn.ModuleList()

		self.model.append(nn.Sequential(nn.Conv2d(1, 16, kernel_size=(2*ncoef,3), stride=(1,1), padding=(0,1), bias=False), nn.BatchNorm2d(16), nn.ReLU()))

		self.model.append(self._make_layer(block, 64, layers[0], stride=1))
		self.model.append(self._make_layer(block, 128, layers[1], stride=2))
		self.model.append(self._make_layer(block, 256, layers[2], stride=2))
		self.model.append(self._make_layer(block, 512, layers[3], stride=2))

		self.initialize_params()

		self.pooling = SelfAttention(block.expansion*512)

		self.post_pooling = nn.Sequential(nn.Conv1d(block.expansion*512*2, 512, 1),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, 512, 1),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, n_z, 1) )

		if proj_size>0 and sm_type!='none':
			if sm_type=='softmax':
				self.out_proj=Softmax(input_features=n_z, output_features=proj_size)
			elif sm_type=='am_softmax':
				self.out_proj=AMSoftmax(input_features=n_z, output_features=proj_size)
			else:
				raise NotImplementedError

		## Load after initialize main model params
		self.encoder = wf_builder(pase_cfg)
		if pase_cp:
			self.encoder.load_pretrained(pase_cp, load_last=True, verbose=False)

	def initialize_params(self):

		for layer in self.modules():
			if isinstance(layer, torch.nn.Conv2d):
				init.kaiming_normal_(layer.weight, a=0, mode='fan_out')
			elif isinstance(layer, torch.nn.Linear):
				init.kaiming_uniform_(layer.weight)
			elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
				layer.weight.data.fill_(1)
				layer.bias.data.zero_()

	def _make_layer(self, block, planes, num_blocks, stride):
		strides = [stride] + [1]*(num_blocks-1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride))
			self.in_planes = planes * block.expansion
		return nn.Sequential(*layers)

	def forward(self, x):

		z = self.encoder(x.unsqueeze(1)).unsqueeze(1)

		z_mu = z.mean(-1, keepdim=True).repeat([1,1,1,z.size(-1)])
		z=torch.cat([z, z_mu], 2)

		for mod_ in self.model:
			z = mod_(z)

		z = z.squeeze(2)

		z = self.pooling(z.permute(0,2,1).contiguous()).unsqueeze(-1)

		z = self.post_pooling(z)

		return z.squeeze()

class ResNet_34(nn.Module):
	def __init__(self, pase_cfg, pase_cp=None, n_z=256, layers=[3,4,6,3], block=PreActBlock, proj_size=0, ncoef=23, sm_type='none'):
		self.in_planes = 16
		super(ResNet_34, self).__init__()

		self.model = nn.ModuleList()

		self.model.append(nn.Sequential(nn.Conv2d(1, 16, kernel_size=(2*ncoef,3), stride=(1,1), padding=(0,1), bias=False), nn.BatchNorm2d(16), nn.ReLU()))

		self.model.append(self._make_layer(block, 64, layers[0], stride=1))
		self.model.append(self._make_layer(block, 128, layers[1], stride=2))
		self.model.append(self._make_layer(block, 256, layers[2], stride=2))
		self.model.append(self._make_layer(block, 512, layers[3], stride=2))

		self.initialize_params()

		self.pooling = SelfAttention(block.expansion*512)

		self.post_pooling = nn.Sequential(nn.Conv1d(block.expansion*512*2, 512, 1),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, 512, 1),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, n_z, 1) )

		if proj_size>0 and sm_type!='none':
			if sm_type=='softmax':
				self.out_proj=Softmax(input_features=n_z, output_features=proj_size)
			elif sm_type=='am_softmax':
				self.out_proj=AMSoftmax(input_features=n_z, output_features=proj_size)
			else:
				raise NotImplementedError

		## Load after initialize main model params
		self.encoder = wf_builder(pase_cfg)
		if pase_cp:
			self.encoder.load_pretrained(pase_cp, load_last=True, verbose=False)

	def initialize_params(self):

		for layer in self.modules():
			if isinstance(layer, torch.nn.Conv2d):
				init.kaiming_normal_(layer.weight, a=0, mode='fan_out')
			elif isinstance(layer, torch.nn.Linear):
				init.kaiming_uniform_(layer.weight)
			elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
				layer.weight.data.fill_(1)
				layer.bias.data.zero_()

	def _make_layer(self, block, planes, num_blocks, stride):
		strides = [stride] + [1]*(num_blocks-1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride))
			self.in_planes = planes * block.expansion
		return nn.Sequential(*layers)

	def forward(self, x):

		z = self.encoder(x.unsqueeze(1)).unsqueeze(1)

		z_mu = z.mean(-1, keepdim=True).repeat([1,1,1,z.size(-1)])
		z=torch.cat([z, z_mu], 2)

		for mod_ in self.model:
			z = mod_(z)

		z = z.squeeze(2)

		z = self.pooling(z.permute(0,2,1).contiguous()).unsqueeze(-1)

		z = self.post_pooling(z)

		return z.squeeze()

class ResNet_18(nn.Module):
	def __init__(self, pase_cfg, pase_cp=None, n_z=256, layers=[2,2,2,2], block=PreActBlock, proj_size=0, ncoef=23, sm_type='none'):
		self.in_planes = 16
		super(ResNet_18, self).__init__()

		self.model = nn.ModuleList()

		self.model.append(nn.Sequential(nn.Conv2d(1, 16, kernel_size=(2*ncoef,3), stride=(1,1), padding=(0,1), bias=False), nn.BatchNorm2d(16), nn.ReLU()))

		self.model.append(self._make_layer(block, 64, layers[0], stride=1))
		self.model.append(self._make_layer(block, 128, layers[1], stride=2))
		self.model.append(self._make_layer(block, 256, layers[2], stride=2))
		self.model.append(self._make_layer(block, 512, layers[3], stride=2))

		self.initialize_params()

		self.pooling = SelfAttention(block.expansion*512)

		self.post_pooling = nn.Sequential(nn.Conv1d(block.expansion*512*2, 512, 1),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, 512, 1),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, n_z, 1) )

		if proj_size>0 and sm_type!='none':
			if sm_type=='softmax':
				self.out_proj=Softmax(input_features=n_z, output_features=proj_size)
			elif sm_type=='am_softmax':
				self.out_proj=AMSoftmax(input_features=n_z, output_features=proj_size)
			else:
				raise NotImplementedError

		## Load after initialize main model params
		self.encoder = wf_builder(pase_cfg)
		if pase_cp:
			self.encoder.load_pretrained(pase_cp, load_last=True, verbose=False)

	def initialize_params(self):

		for layer in self.modules():
			if isinstance(layer, torch.nn.Conv2d):
				init.kaiming_normal_(layer.weight, a=0, mode='fan_out')
			elif isinstance(layer, torch.nn.Linear):
				init.kaiming_uniform_(layer.weight)
			elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
				layer.weight.data.fill_(1)
				layer.bias.data.zero_()

	def _make_layer(self, block, planes, num_blocks, stride):
		strides = [stride] + [1]*(num_blocks-1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride))
			self.in_planes = planes * block.expansion
		return nn.Sequential(*layers)

	def forward(self, x):

		z = self.encoder(x.unsqueeze(1)).unsqueeze(1)

		z_mu = z.mean(-1, keepdim=True).repeat([1,1,1,z.size(-1)])
		z=torch.cat([z, z_mu], 2)

		for mod_ in self.model:
			z = mod_(z)

		z = z.squeeze(2)

		z = self.pooling(z.permute(0,2,1).contiguous()).unsqueeze(-1)

		z = self.post_pooling(z)

		return z.squeeze()

class StatisticalPooling(nn.Module):

	def forward(self, x):
		# x is 3-D with axis [B, feats, T]
		mu = x.mean(dim=2, keepdim=True)
		std = x.std(dim=2, keepdim=True)
		return torch.cat((mu, std), dim=1)

class TDNN(nn.Module):
	# Architecture taken from https://github.com/santi-pdp/pase/blob/master/pase/models/tdnn.pyf
	def __init__(self, pase_cfg, pase_cp=None, n_z=256, proj_size=0, ncoef=100, sm_type='none'):
		super(TDNN, self).__init__()

		self.encoder = wf_builder(pase_cfg)
		if pase_cp:
			self.encoder.load_pretrained(pase_cp, load_last=True, verbose=False)

		self.model = nn.Sequential( nn.BatchNorm1d(2*ncoef),
			nn.Conv1d(2*ncoef, 512, 5, padding=2),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, 512, 3, dilation=2, padding=2),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, 512, 3, dilation=3, padding=3),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, 512, 1),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, 1500, 1),
			nn.BatchNorm1d(1500),
			nn.ReLU(inplace=True))

		self.pooling = StatisticalPooling()

		self.post_pooling = nn.Sequential(nn.Conv1d(3000, 512, 1),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, 512, 1),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, n_z, 1) )

		if proj_size>0 and sm_type!='none':
			if sm_type=='softmax':
				self.out_proj=Softmax(input_features=n_z, output_features=proj_size)
			elif sm_type=='am_softmax':
				self.out_proj=AMSoftmax(input_features=n_z, output_features=proj_size)
			else:
				raise NotImplementedError

	def forward(self, x):

		z = self.encoder(x.unsqueeze(1))

		z_mu = z.mean(-1, keepdim=True).repeat([1,1,z.size(-1)])
		z=torch.cat([z, z_mu], 1)

		z = self.model(z)
		z = self.pooling(z)
		z = self.post_pooling(z)

		return z.squeeze()

class TDNN_mfcc(nn.Module):
	# Architecture taken from https://github.com/santi-pdp/pase/blob/master/pase/models/tdnn.pyf
	def __init__(self, n_z=256, proj_size=0, ncoef=100, sm_type='none'):
		super(TDNN_mfcc, self).__init__()

		self.model = nn.Sequential( nn.BatchNorm1d(ncoef),
			nn.Conv1d(ncoef, 512, 5, padding=2),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, 512, 3, dilation=2, padding=2),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, 512, 3, dilation=3, padding=3),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, 512, 1),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, 1500, 1),
			nn.BatchNorm1d(1500),
			nn.ReLU(inplace=True))

		self.pooling = StatisticalPooling()

		self.post_pooling = nn.Sequential(nn.Conv1d(3000, 512, 1),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, 512, 1),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, n_z, 1) )

		if proj_size>0 and sm_type!='none':
			if sm_type=='softmax':
				self.out_proj=Softmax(input_features=n_z, output_features=proj_size)
			elif sm_type=='am_softmax':
				self.out_proj=AMSoftmax(input_features=n_z, output_features=proj_size)
			else:
				raise NotImplementedError

	def forward(self, x):

		z = self.model(x)
		z = self.pooling(z)
		z = self.post_pooling(z)

		return z.squeeze(-1)

class MLP(nn.Module):
	def __init__(self, pase_cfg, pase_cp=None, n_z=256, proj_size=0, ncoef=100, sm_type='none'):
		super(MLP, self).__init__()

		self.encoder = wf_builder(pase_cfg)
		if pase_cp:
			self.encoder.load_pretrained(pase_cp, load_last=True, verbose=False)

		self.model = nn.Sequential( nn.Conv1d(2*ncoef, 512, 1),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, 1500, 1),
			nn.BatchNorm1d(1500),
			nn.ReLU(inplace=True))

		self.pooling = StatisticalPooling()

		self.post_pooling = nn.Sequential(nn.Conv1d(3000, 512, 1),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, 512, 1),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, n_z, 1) )

		if proj_size>0 and sm_type!='none':
			if sm_type=='softmax':
				self.out_proj=Softmax(input_features=n_z, output_features=proj_size)
			elif sm_type=='am_softmax':
				self.out_proj=AMSoftmax(input_features=n_z, output_features=proj_size)
			else:
				raise NotImplementedError

	def forward(self, x):

		z = self.encoder(x.unsqueeze(1))

		z_mu = z.mean(-1, keepdim=True).repeat([1,1,z.size(-1)])
		z=torch.cat([z, z_mu], 1)

		z = self.model(z)
		z = self.pooling(z)
		z = self.post_pooling(z)

		return z.squeeze()

class global_MLP(nn.Module):
	def __init__(self, pase_cfg, pase_cp=None, n_z=256, proj_size=0, ncoef=100, sm_type='none'):
		super(global_MLP, self).__init__()

		self.encoder = wf_builder(pase_cfg)
		if pase_cp:
			self.encoder.load_pretrained(pase_cp, load_last=True, verbose=False)

		self.model = nn.Sequential(nn.Linear(ncoef, 512),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Linear(512, 512),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Linear(512, n_z) )

		if proj_size>0 and sm_type!='none':
			if sm_type=='softmax':
				self.out_proj=Softmax(input_features=n_z, output_features=proj_size)
			elif sm_type=='am_softmax':
				self.out_proj=AMSoftmax(input_features=n_z, output_features=proj_size)
			else:
				raise NotImplementedError

	def forward(self, x):

		z = self.encoder(x.unsqueeze(1)).mean(-1)
		z = self.model(z)

		return z

class pyr_rnn(nn.Module):
	def __init__(self, pase_cfg, pase_cp=None, n_layers=4, n_z=256, proj_size=0, ncoef=23, sm_type='none'):
		super(pyr_rnn, self).__init__()

		self.model = nn.ModuleList([nn.LSTM(2*ncoef, 256, 1, bidirectional=True, batch_first=True)])

		for i in range(1,n_layers):
			self.model.append(nn.LSTM(256*2*2, 256, 1, bidirectional=True, batch_first=True))

		self.pooling = StatisticalPooling()

		self.post_pooling = nn.Sequential(nn.Conv1d(256*2*2*2, 512, 1),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, 512, 1),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, n_z, 1) )

		self.initialize_params()

		self.attention = SelfAttention(512)

		if proj_size>0 and sm_type!='none':
			if sm_type=='softmax':
				self.out_proj=Softmax(input_features=n_z, output_features=proj_size)
			elif sm_type=='am_softmax':
				self.out_proj=AMSoftmax(input_features=n_z, output_features=proj_size)
			else:
				raise NotImplementedError

		self.encoder = wf_builder(pase_cfg)
		if pase_cp:
			self.encoder.load_pretrained(pase_cp, load_last=True, verbose=False)

	def initialize_params(self):

		for layer in self.modules():
			if isinstance(layer, torch.nn.Conv2d):
				init.kaiming_normal_(layer.weight, a=0, mode='fan_out')
			elif isinstance(layer, torch.nn.Linear):
				init.kaiming_uniform_(layer.weight)
			elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
				layer.weight.data.fill_(1)
				layer.bias.data.zero_()

	def _make_layer(self, block, planes, num_blocks, stride):
		strides = [stride] + [1]*(num_blocks-1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride))
			self.in_planes = planes * block.expansion
		return nn.Sequential(*layers)

	def forward(self, x, inner=False):

		x = self.encoder(x.unsqueeze(1))

		x = x.permute(0,2,1)

		batch_size = x.size(0)
		seq_size = x.size(1)

		x_mu = x.mean(1, keepdim=True).repeat([1,seq_size,1])

		x=torch.cat([x, x_mu], -1)

		h0 = torch.zeros(2, batch_size, 256)
		c0 = torch.zeros(2, batch_size, 256)

		if x.is_cuda:
			h0 = h0.cuda(x.get_device())
			c0 = c0.cuda(x.get_device())

		for mod_ in self.model:
			x, (h_, c_) = mod_(x, (h0, c0))
			if x.size(1)%2>0:
				x=x[:,:-1,:]
			x = x.contiguous().view(x.size(0), -1, x.size(-1)*2)

		x = self.pooling(x.transpose(1,2))

		x = self.post_pooling(x)

		return x.squeeze()
