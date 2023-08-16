#!/usr/bin/env python3

import torch
from torch.nn import Linear, L1Loss
import torch.nn.functional as F
import scipy.stats
import numpy as np
from torch_geometric_temporal.nn.recurrent import GConvLSTM, GConvGRU
from torch_geometric_temporal.signal import temporal_signal_split
from time import time
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RecurrentGCN_LSTM(torch.nn.Module):
	def __init__(self, node_features, hidden_channels=32, K=1):
		super(RecurrentGCN_LSTM, self).__init__()
		self.recurrent = GConvLSTM(node_features, hidden_channels, K)
		self.linear = torch.nn.Linear(hidden_channels, 1)

	def forward(self, x, edge_index, edge_weight, h, c):
		h_0, c_0 = self.recurrent(x, edge_index, edge_weight, h, c)
		h = F.relu(h_0)
		h = self.linear(h)
		return h, h_0, c_0

class RecurrentGCN_GRU(torch.nn.Module):
	def __init__(self, node_features, hidden_channels=32, K=1):
		super(RecurrentGCN_GRU, self).__init__()
		self.recurrent = GConvGRU(node_features, hidden_channels, K)
		self.linear = torch.nn.Linear(hidden_channels, 1)

	def forward(self, x, edge_index, edge_weight, h, c):
		h = self.recurrent(x, edge_index, edge_weight)
		h = F.relu(h)
		h = self.linear(h)
		return h, None, None

def create_model(args, dataset):
	node_features = dataset.features[0].shape[1]
	if args.recurrent_layer == 'LSTM':
		cls = RecurrentGCN_LSTM
	else:
		cls = RecurrentGCN_GRU

	model = cls(node_features=node_features, hidden_channels=args.hidden_channels, K=args.k).to(device)

	return model

def enumerate_to_device(dataset):
	'Enumerate the dataset temporal snapshots and send the tensors to device'

	for time, snapshot in enumerate(dataset):
		snapshot.x = snapshot.x.to(device)
		snapshot.edge_index = snapshot.edge_index.to(device)
		snapshot.edge_attr = snapshot.edge_attr.to(device)
		snapshot.y = snapshot.y.to(device)

		yield time, snapshot

def train(model, optimizer, dataset):
	model.train()
	optimizer.zero_grad()
	loss = 0
	h, c = None, None
	for time, snapshot in enumerate_to_device(dataset):
		y_hat, h, c = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr, h, c)
		loss += F.mse_loss(y_hat, snapshot.y)
	loss /= time + 1
	loss.backward()
	optimizer.step()
	return float(loss), (h, c)

def test(model, optimizer, dataset, *pass_args):
	model.eval()
	loss = 0
	mape_loss = 0
	for time, snapshot in enumerate_to_device(dataset):
		y_hat, h, c = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr, *pass_args)
		loss += F.mse_loss(y_hat, snapshot.y)
		mape_loss += ((y_hat - snapshot.y) / (snapshot.y + 0.000001)).abs().mean()
	loss /= time + 1
	mape_loss /= time + 1
	return float(loss), float(mape_loss)

def trainable_parameter_count(model):
	model_parameters = filter(lambda p: p.requires_grad, model.parameters())
	return sum([np.prod(p.size()) for p in model_parameters])

def print_training_info(args, model):
	# print only once
	if hasattr(print_training_info, 'done') and print_training_info.done:
		return

	print(f'Training on {torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU"}\n')
	print(f'Hidden channels: {args.hidden_channels}')
	print(f'Convolution = GConv{args.recurrent_layer}(K={args.k})\n')
	print(f'Number of trainable model parameters: {trainable_parameter_count(model)}\n')

	print_training_info.done = True

def run(args, train_dataset, test_dataset, run_number=None):
	model = create_model(args, train_dataset)
	print_training_info(args, model)
	optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

	run_str = f'Run {run_number} ' if run_number is not None else ''

	record = []

	beg = time()

	for epoch in range(args.epochs):
		beg_epoch = time()

		train_loss, pass_args = train(model, optimizer, train_dataset)
		test_loss, mape = test(model, optimizer, test_dataset, *pass_args)

		end_epoch = time()

		train_rmse = train_loss**0.5
		test_rmse = test_loss**0.5

		print(f'{run_str}Epoch: {epoch+1:03d}, Train RMSE: {train_rmse:.6f}, Test RMSE: {test_rmse:.6f}, MAPE: {mape:.6f}, in {end_epoch-beg_epoch:.02f}s', flush=True)

		record.append((train_rmse, test_rmse, mape))

	end = time()

	print(f'\n{run_str}Trained {args.epochs} epochs in {end-beg:.02f}s\n')

	return record, model

def mean_and_conf_interval(vals, confidence_interval=0.99):
	'Compute the mean and confidence interval'

	n = len(vals)

	double_sided_quantil = 1.0 - (1.0 - confidence_interval) / 2.0
	conf_int_mul = scipy.stats.t.ppf(double_sided_quantil, df=n - 1)

	mean = sum(vals) / n
	mean_std = (sum([(val - mean)**2 for val in vals]) / (n * (n - 1)))**0.5
	lo = mean - conf_int_mul * mean_std
	hi = mean + conf_int_mul * mean_std

	return f'{lo:.8f} {mean:.8f} {hi:.8f}'

def predict(model, dataset, train_dataset, test_dataset, first_n):
	n = test_dataset[-1].y.shape[0]
	# compute the permutation which will give us sorting by last score
	permute = torch.sort(test_dataset[-1].y, dim=0, descending=True).indices

	# first print the expected scores
	for time, snapshot in enumerate(dataset):
		lst = snapshot.y[permute].reshape(n).tolist()[:first_n]
		print(f'{time}\t' + '\t'.join([f'{x:.8f}' for x in lst]))

	# empty line
	print()

	# go through train interval
	h, c = None, None
	for time, snapshot in enumerate_to_device(train_dataset):
		y_hat, h, c = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr, h, c)

	# print predictions for test interval
	for time, snapshot in enumerate_to_device(test_dataset):
		y_hat, h, c = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr, h, c)
		lst = y_hat[permute].reshape(n).tolist()[:first_n]
		print(f'{time+len(train_dataset.features)}\t' + '\t'.join([f'{x:.8f}' for x in lst]))

parser = argparse.ArgumentParser(description='Hotel eigenvector centrality temporal prediction executor')
parser.add_argument('-i', '--input', type=str, metavar='FILE', required=True,
		    help='dataset input FILE, created by create_torch_data.py')
parser.add_argument('-e', '--epochs', type=int, metavar='N', default=400,
		    help='number of epochs to train for (default: 400)')
parser.add_argument('-l', '--layers', type=int, metavar='N', default=2,
		    help='number of convolutional layers (default: 2)')
parser.add_argument('--hidden-channels', type=int, metavar='N', default=8,
		    help='number of hidden channels per layer (default: 8)')
parser.add_argument('--recurrent-layer', choices=['GRU', 'LSTM'], default='LSTM',
		    help='use CONV as recurrent layer (default: LSTM)')
parser.add_argument('--k', type=int, metavar='K', default=3,
		    help='use K as degree of the Chebyshev polynomial (default: 3)')
parser.add_argument('--repeat', type=int, metavar='N', default=1,
		    help='run training N times and compute averages and confidence intervals (default: 1)')
parser.add_argument('--learning-rate', type=float, default=0.001,
		    help='learning rate for the Adam optimizer')
parser.add_argument('--predict-with-saved-model', type=str, metavar='FILE',
		    help='load previously saved model from FILE and dump predictions')
parser.add_argument('--save-best-model', type=str, metavar='FILE',
		    help='save best model to FILE')

def main():
	args = parser.parse_args()

	dataset = torch.load(args.input)

	# remove last 72 months
	dataset = dataset[:dataset.snapshot_count - 72]
	train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)

	if args.predict_with_saved_model:
		model = torch.load(args.predict_with_saved_model)
		model.to(device)
		predict(model, dataset, train_dataset, test_dataset, 10)
		exit(0)

	if args.repeat > 1:
		records = []
		models = []

		# run several trainings and save models and losses across epochs
		for i in range(args.repeat):
			record, model = run(args, dataset, train_dataset, test_dataset, i + 1)
			records.append(record)
			models.append(model.to('cpu'))

		# compute mean losses per epoch
		for epoch in range(args.epochs):
			train_rmse = mean_and_conf_interval([ records[i][epoch][0] for i in range(args.repeat) ])
			test_rmse = mean_and_conf_interval([ records[i][epoch][1] for i in range(args.repeat) ])
			test_mape = mean_and_conf_interval([ records[i][epoch][2] for i in range(args.repeat) ])
			
			print(f'Epoch: {epoch+1:03d}   TrainRMSE= {train_rmse}   TestRMSE= {test_rmse}   TestMAPE= {test_mape}')

		# find best model (with min test loss at the last epoch)
		best_model = models[min(range(args.repeat), key=lambda i: records[i][args.epochs - 1][1])]
	else:
		record, best_model = run(args, train_dataset, test_dataset)

	if args.save_best_model:
		torch.save(best_model, args.save_best_model)
		print(f'Best model saved to {args.save_best_model}')

if __name__ == '__main__':
	main()
