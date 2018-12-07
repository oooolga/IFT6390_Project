import torch
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()

import argparse
import os
import numpy as np
from tqdm import tqdm 
import pdb

from util.load_data import load_data
from util.model_util import save_checkpoint, load_checkpoint
from models import *

result_path = 'results/'

state = {'train_loss': [],
		 'train_acc': [],
		 'valid_loss': [],
		 'valid_acc': [],
		 'test_loss': [],
		 'test_acc': []}

models = {'CNN': CNNModel,
		  'Regression': RegressionModel}

def load_model(model_name, dataset_name):
	'''
	This function returns model by name.
	'''
	model_kwargs = set_model_kwargs(model_name, dataset_name)
	return models[model_name](**model_kwargs)

def set_model_kwargs(model_name, dataset_name):
	'''
	This function sets up model input argument.
	'''
	if model_name == 'CNN':
		if dataset_name == 'CIFAR':
			return {'c_in': 3,
					'nlabels': 10}
		if dataset_name == 'EMNIST':
			return {'c_in': 1,
					'nlabels': 47}
		if dataset_name == 'FMNIST':
			return {'c_in': 1,
					'nlabels': 10}
	if model_name == 'Regression':
		# add code here
		if dataset_name == 'FMNIST':
			return {'c_in': 1,
					'input_size': 28,
					'nlabels': 10}
		if dataset_name == 'EMNIST':
			return {'c_in': 1,
					'input_size': 28,
					'nlabels': 47}
		if dataset_name == 'CIFAR':
			return {'c_in': 3,
					'input_size': 32,
					'nlabels': 10}
		#pass
	if model_name == 'NN':
		# TODO
		raise NotImplementedError
	return {}

def parse():
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', default='CIFAR', type=str,
						choices=['CIFAR', 'FMNIST', 'EMNIST'], help='Dataset choice.')
	parser.add_argument('--model', default='CNN', type=str,
						choices=['CNN', 'NN', 'Regression'], help='Model type.')
	parser.add_argument('--load_model', required=True, type=str, help='Load model path.')
	parser.add_argument('--batch_size', default=200, type=int,
						help='Mini-batch size for testing.')

	args = parser.parse_args()
	return args


def test(model, data_loader, mode='valid'):
	model.eval()

	loss_avg = 0.0
	correct = 0

	with torch.no_grad():
		for i_batch, batch in tqdm(enumerate(data_loader)):
			data, target = batch

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

	# load dataset
	train_loader, valid_loader, test_loader = \
			load_data(args.batch_size, args.batch_size, args.dataset)

	# load model
	model = load_model(args.model, args.dataset)
	if use_cuda:
		model.cuda()

	# count total number of parameters
	model_parameters = filter(lambda p: p.requires_grad, model.parameters())
	params = sum([np.prod(p.size()) for p in model_parameters])
	print('\nTotal number of parameters: {}\n'.format(params))

	model, optimizer, epoch_start, best_valid_acc = \
							load_checkpoint(args.load_model, model, None)

	test(model, train_loader, mode='train')
	test(model, valid_loader)
	test(model, test_loader, mode='test')

	# output result
	print('|\t\t[Train]:\taccuracy={:.3f}\tloss={:.3f}'.format(state['train_acc'][-1],
															   state['train_loss'][-1]))
	print('|\t\t[Valid]:\taccuracy={:.3f}\tloss={:.3f}'.format(state['valid_acc'][-1],
															   state['valid_loss'][-1]))
	print('|\t\t[Test]:\taccuracy={:.3f}\tloss={:.3f}'.format(state['test_acc'][-1],
															  state['test_loss'][-1]))
