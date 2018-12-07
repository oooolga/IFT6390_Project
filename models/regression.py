import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

import pdb

class RegressionModel(nn.Module):
	def __init__(self, c_in=3, input_size=32, nlabels=10): # you might need to change function argument
		super(RegressionModel, self).__init__()
		# add code here
		self.nlabels = nlabels
		self.input_size = input_size
		self.linear = nn.Linear(input_size*input_size*c_in, nlabels)

	def forward(self, x): # you might need to change function argument
		# add code here
		batch_size = x.size(0)
		x = x.view(batch_size, -1)
		out = self.linear(x)
		return out