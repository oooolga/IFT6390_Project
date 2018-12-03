import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

import pdb

class NNModel(nn.Module):
	def __init__(self, d_in=1024, nlabels=10):
		super(NNModel, self).__init__()
		self.nlabels = nlabels
		self.d_in = d_in


		self.mlp = nn.Sequential(
			nn.Linear(self.d_in, int(self.d_in*0.5)),
			nn.ReLU(),
			nn.Linear(int(self.d_in*0.5), 300),
			nn.ReLU(),
			nn.Linear(300, 100),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(100, self.nlabels)
			)

		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.kaiming_normal_(m.weight)

	def forward(self, x):
		batch_size = x.size(0)
		x = x.view(batch_size, -1)
		return self.mlp(x)