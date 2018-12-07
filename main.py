import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

use_cuda = torch.cuda.is_available()

import argparse
import os
import numpy as np
from tqdm import tqdm 
import pdb

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from util.load_data import load_data
from util.model_util import save_checkpoint, load_checkpoint
from models import *

state = {'train_loss': [],
		 'train_acc': [],
		 'valid_loss': [],
		 'valid_acc': []}
plot_state = {'train_loss': [],
			  'train_acc': [],
			  'valid_loss': [],
			  'valid_acc': [],
			  'epochs': []}
models = {'CNN': CNNModel,
		  'Regression': RegressionModel,
		  'NN': NNModel,
		  'PolyRegression' : PolyRegressionModel}

model_path = 'saved_models/'
result_path = 'results/'

#polynome degree for Polynomial Regression
ndegree = 4


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
	if model_name == 'PolyRegression':
		if dataset_name == 'FMNIST':
			return {'c_in': 1,
					'input_size': 28,
					'nlabels': 10,
					'ndegree': ndegree}
		if dataset_name == 'EMNIST':
			return {'c_in': 1,
					'input_size': 28,
					'nlabels': 47,
					'ndegree': ndegree}
		if dataset_name == 'CIFAR':
			return {'c_in': 3,
					'input_size': 32,
					'nlabels': 10,
					'ndegree': ndegree}

	if model_name == 'NN':
		if dataset_name == 'CIFAR':
			return {'d_in': 1024*3,
					'nlabels': 10}
		if dataset_name == 'EMNIST':
			return {'d_in': 784,
					'nlabels': 47}
		if dataset_name == 'FMNIST':
			return {'d_in':784,
					'nlabels': 10}
	return {}

def parse():
	parser = argparse.ArgumentParser()
	parser.add_argument('-lr', '--learning_rate', default=5e-2, type=float,
						help='Learning rate.')
	parser.add_argument('--batch_size', default=50, type=int,
						help='Mini-batch size for training.')
	parser.add_argument('--test_batch_size', default=200, type=int,
						help='Mini-batch size for testing.')
	parser.add_argument('--epochs', default=100, type=int,
						help='Total number of epochs.')
	parser.add_argument('--seed', default=123, type=int,
						help='Random number seed.')
	parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay.')
	parser.add_argument('--model_name', required=True, type=str, help='Model name.')
	parser.add_argument('--load_model', default=None, type=str, help='Load model path.')
	parser.add_argument('--optimizer', default='Adam', type=str, choices=['Adam', 'SGD'],
						help='Optimizer type.')
	parser.add_argument('--dataset', default='CIFAR', type=str,
						choices=['CIFAR', 'FMNIST', 'EMNIST'], help='Dataset choice.')
	parser.add_argument('--model', default='CNN', type=str,
						choices=['CNN', 'NN', 'Regression','PolyRegression'], help='Model type.')
	parser.add_argument('--plot_freq', default=5, type=int,
						help='plot_freq')
	parser.add_argument('--pdegree', default=4, type=int,
						help='polynome degree')

	args = parser.parse_args()
	return args

def print_model_setting(args):
	print('Model type: {}'.format(args.model))
	if args.model == 'PolyRegression':
		print('polynome degree: {}'.format(args.pdegree))
	print('Dataset: {}'.format(args.dataset))
	print('Optimizer type: {}'.format(args.optimizer))
	print('Learning rate: {}'.format(args.learning_rate))
	print('Total number of epochs: {}'.format(args.epochs))
	print('Learning rate: {}'.format(args.learning_rate))
	print('Weight decay: {}'.format(args.weight_decay))
	print('Batch size: {}'.format(args.batch_size))
	print('Plot frequency: {}\n'.format(args.plot_freq))

def train(model, optimizer, train_loader):
	model.train()

	for i_batch, batch in tqdm(enumerate(train_loader)):

		data, target = batch

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
	ndegree= args.pdegree;

	# set seeds
	torch.manual_seed(args.seed)
	if use_cuda:
		torch.cuda.manual_seed_all(args.seed)

	# load dataset
	train_loader, valid_loader, test_loader = \
			load_data(args.batch_size, args.test_batch_size, args.dataset)

	# load model
	model = load_model(args.model, args.dataset)
	if use_cuda:
		model.cuda()

	# count total number of parameters
	model_parameters = filter(lambda p: p.requires_grad, model.parameters())
	params = sum([np.prod(p.size()) for p in model_parameters])
	print('\nTotal number of parameters: {}\n'.format(params))

	print_model_setting(args)

	# set optimizer
	if args.optimizer == 'Adam':
		optimizer = optim.Adam(params=model.parameters(), lr=args.learning_rate,
							   weight_decay=args.weight_decay)
	else:
		optimizer = optim.SGD(model.parameters(), lr = args.learning_rate, momentum=0.9,
							  weight_decay=args.weight_decay)

	# set scheduler
	scheduler = StepLR(optimizer, step_size=10, gamma=0.8)

	# load model if needed
	if args.load_model is None:
		epoch_start = -1
		best_valid_acc = 0
	else:
		model, optimizer, epoch_start, best_valid_acc = \
								load_checkpoint(args.load_model, model, optimizer)

	# iterative learning
	for epoch_i in range(epoch_start+1, args.epochs+1):
		print('|\tEpoch {}/{}:'.format(epoch_i, args.epochs))
		scheduler.step()

		# train
		if epoch_i != 0:
			train(model, optimizer, train_loader)
		# evaluate
		test(model, train_loader, mode='train')
		test(model, valid_loader)

		# output result
		print('|\t\t[Train]:\taccuracy={:.3f}\tloss={:.3f}'.format(state['train_acc'][-1],
																   state['train_loss'][-1]))
		print('|\t\t[Valid]:\taccuracy={:.3f}\tloss={:.3f}'.format(state['valid_acc'][-1],
																   state['valid_loss'][-1]))

		# add plot results
		if epoch_i%args.plot_freq == 0:
			for k in state:
				plot_state[k].append(state[k][-1])
			plot_state['epochs'].append(epoch_i)

		# saving model
		if state['valid_acc'][-1] > best_valid_acc:
			best_valid_acc = state['valid_acc'][-1]
			save_checkpoint({
				'epoch_i': epoch_i,
				'state_dict': model.state_dict(),
				'optimizer': optimizer.state_dict(),
				'best_acc': best_valid_acc
				}, os.path.join(model_path, args.model_name+'.pt'))


	# plot training curves
	plt.close('all')
	fig, ax1 = plt.subplots()
	line_ta = ax1.plot(plot_state['epochs'], plot_state['train_acc'], color="#7aa0c4", label='train acc')
	line_va = ax1.plot(plot_state['epochs'], plot_state['valid_acc'], color="#ca82e1", label='valid acc')
	ax1.set_xlabel('epoch')
	ax1.set_ylabel('accuracy')

	ax2 = ax1.twinx()
	line_tl = ax2.plot(plot_state['epochs'], plot_state['train_loss'], color="#8bcd50", label='train loss')
	line_vl = ax2.plot(plot_state['epochs'], plot_state['valid_loss'], color="#e18882", label='valid loss')
	ax2.set_ylabel('loss')

	lines = line_ta + line_va + line_tl + line_vl
	labs = [l.get_label() for l in lines]

	#fig.subplots_adjust(right=0.75) 
	box = ax1.get_position()
	ax1.set_position([box.x0, box.y0 + box.height * 0.1,
					box.width, box.height * 0.9])
	ax1.legend(lines, labs, loc='upper center', bbox_to_anchor=(0.5, -0.05),
     				 fancybox=True, shadow=True, ncol=5)
	fig.tight_layout()
	plt.title('Training curves')
	plt.savefig(os.path.join(result_path, args.model_name+'_training_curve.png'),
				bbox_inches='tight')
	plt.clf()
	plt.close('all')
