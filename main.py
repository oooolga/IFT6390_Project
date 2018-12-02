import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()

import argparse
import os
import numpy as np
from tqdm import tqdm 
from models.basic_model import Model
import pdb

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from util.load_data import load_cifar100_data

def parse():
	parser = argparse.ArgumentParser()
	parser.add_argument('-lr', '--learning_rate', default=5e-2, type=float,
						help='Learning rate.')
	parser.add_argument('--batch_size', default=50, type=int,
						help='Mini-batch size for training.')
	parser.add_argument('--test_batch_size', default=200, type=int,
						help='Mini-batch size for testing.')
	parser.add_argument('--epochs', default=200, type=int,
						help='Total number of epochs.')
	parser.add_argument('--seed', default=123, type=int,
						help='Random number seed.')
	parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay.')
	parser.add_argument('--model_name', required=True, type=str, help='Model name.')
	parser.add_argument('--load_model', default=None, type=str, help='Load model path.')
	parser.add_argument('--optimizer', default='Adam', type=str, help='Optimizer type.')
	parser.add_argument('--load_all_train', action='store_true', help='Load all data as train flag.')
	parser.add_argument('--plot_path', default='results', type=str, help='Path for plots.')

	args = parser.parse_args()
	return args

if __name__ == '__main__':
	# get arguments
	args = parse()

	# set seeds
	torch.manual_seed(args.seed)
	if use_cuda:
		torch.cuda.manual_seed_all(args.seed)