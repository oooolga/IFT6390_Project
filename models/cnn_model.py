import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

import pdb

class CNNModel(nn.Module):
	def __init__(self, c_in=1, nlabels=10):
		super(CNNModel, self).__init__()
		self.nlabels = nlabels
		self.c_in = c_in
		self.features = nn.Sequential(
			nn.Conv2d(c_in, 64, kernel_size=3, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2),
			nn.Conv2d(64, 64, kernel_size=5),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 128, kernel_size=5),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 128, kernel_size=5),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 128, kernel_size=3),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
			nn.Dropout2d(),
			nn.AdaptiveAvgPool2d((1,1)))
		self.classifier = nn.Linear(128, nlabels)

		nn.init.kaiming_normal_(self.classifier.weight)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def forward(self, x):
		batch_size = x.size(0)
		out = self.features(x).view(batch_size, -1)
		out = self.classifier(out)
		return out