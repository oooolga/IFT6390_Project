import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

import pdb

class PolyRegressionModel(nn.Module):
	def __init__(self, c_in=3, input_size=32, nlabels=10, ndegree=4): # you might need to change function argument
		super(PolyRegressionModel, self).__init__()
		self.nlabels = nlabels
		self.input_size = input_size
		self.ndegree = ndegree;
		self.linear = nn.Linear(input_size*input_size*c_in*ndegree, nlabels)

	def forward(self, x): # you might need to change function argument
		"""Builds features i.e. a matrix with columns [x, x^2, x^3, x^4]."""
		x = x.unsqueeze(1)
		x= torch.cat([x ** i for i in range(1, self.ndegree+1)], 1)
		
		batch_size = x.size(0)
		x = x.view(batch_size, -1)
		out = self.linear(x)
		return out