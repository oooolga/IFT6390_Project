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

from util.load_data import load_data

state = {'train_loss': [],
		 'train_acc': [],
		 'valid_loss': [],
		 'valid_acc': []}
plot_state = {'train_loss': [],
			  'train_acc': [],
			  'valid_loss': [],
			  'valid_acc': [],
			  'epochs': []}

model_path = 'saved_models/'
result_path = 'results/'

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
	parser.add_argument('--optimizer', default='Adam', type=str, choices=['Adam', 'SGD'],
						help='Optimizer type.')
	parser.add_argument('--load_all_train', action='store_true', help='Load all data as train flag.')
	parser.add_argument('--dataset', default='CIFAR', type=str,
						choices=['CIFAR', 'FMNIST', 'EMNIST'], help='Dataset choice.')
	parser.add_argument('--model', default='CNN', type=str,
						choices=['CNN', 'NN', 'Regression'], help='Model type.')

	args = parser.parse_args()
	return args

def train(model, optimizer, train_loader):
	model.train()

	for i_batch, batch in tqdm(enumerate(train_loader)):

		data, target = batch['image'].type(torch.FloatTensor), \
					   batch['label'].type(torch.long)

		if use_cuda:
			data, target = data.cuda(), target.cuda()

		output = model(data)
		optimizer.zero_grad()
		loss = F.cross_entropy(output, target)
		loss.backward()
		optimizer.step()


def test(model, data_loader, mode='valid'):
	model.eval()

	loss_avg = 0.0
	correct = 0

	with torch.no_grad():
		for i_batch, batch in tqdm(enumerate(data_loader)):

			data, target = batch['image'].type(torch.FloatTensor), \
						   batch['label'].type(torch.long)

			if use_cuda:
				data, target = data.cuda(), target.cuda()

			output = model(data)
			loss = F.cross_entropy(output, target)

			pred = output.data.max(1)[1]
			correct += float(pred.eq(target.data).sum())

			loss_avg += float(loss)

			del output, data, target

	state['{}_loss'.format(mode)].append(loss_avg / len(data_loader))
	state['{}_acc'.format(mode)].append(correct / len(data_loader.dataset))


if __name__ == '__main__':
	# get arguments
	args = parse()

	# set seeds
	torch.manual_seed(args.seed)
	if use_cuda:
		torch.cuda.manual_seed_all(args.seed)

	model = VGGModel(vgg_name='VGG13')
	if use_cuda:
		model.cuda()

	model_parameters = filter(lambda p: p.requires_grad, model.parameters())
	params = sum([np.prod(p.size()) for p in model_parameters])
	print('Total number of parameters: {}\n'.format(params))

	if args.optimizer == 'Adam':

		optimizer = optim.Adam(params=model.parameters(), lr=args.learning_rate,
							   weight_decay=args.weight_decay)
	else:
		optimizer = optim.SGD(model.parameters(), lr = args.learning_rate, momentum=0.9,
							  weight_decay=args.weight_decay)

	scheduler = StepLR(optimizer, step_size=10, gamma=0.8)

	data_loaders, in_channels = load_data(args.batch_size, args.test_batch_size, args.dataset)
	train_loader, valid_loader, test_loader = data_loaders

	for epoch_i in range(epoch_start, args.epochs+1):
		print('|\tEpoch {}/{}:'.format(epoch_i, args.epochs))
		scheduler.step()

		if epoch_i != 0:
			train(model, optimizer, train_loader)

		test(model, train_loader, mode='train')
		test(model, valid_loader)

		print('|\t\t[Train]:\taccuracy={:.3f}\tloss={:.3f}'.format(state['train_acc'][-1],
																   state['train_loss'][-1]))
		print('|\t\t[Valid]:\taccuracy={:.3f}\tloss={:.3f}'.format(state['valid_acc'][-1],
																   state['valid_loss'][-1]))