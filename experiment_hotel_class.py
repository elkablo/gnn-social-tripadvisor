#!/usr/bin/env python3

import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.data import HeteroData, InMemoryDataset
from torch_geometric.nn import SAGEConv, to_hetero, GatedGraphConv, GCNConv
import scipy.stats
import torch_geometric.transforms as T
import pandas as pd
import numpy as np
from prediction_models import NodeLabelPredModel, hetero_edge_conv_map, homo_edge_conv_map
from time import time
import argparse
import pickle

parser = argparse.ArgumentParser(description='Hotel label prediction executor')
parser.add_argument('-i', '--input', type=str, metavar='FILE', required=True,
		    help='dataset input FILE, created by create_torch_data.py')
parser.add_argument('-k', '--k-folds', type=int, metavar='K', default=10,
		    help='k for k-cross validation (default: 10)')
parser.add_argument('-e', '--epochs', type=int, metavar='N', default=400,
		    help='number of epochs to train for (default: 400)')
parser.add_argument('-l', '--layers', type=int, metavar='N', default=2,
		    help='number of convolutional layers (default: 2)')
parser.add_argument('--hidden-channels', type=int, metavar='N', default=8,
		    help='number of hidden channels per layer (default: 8)')
parser.add_argument('--hetero-edge-convolution', choices=hetero_edge_conv_map.keys(), default='SAGE',
		    help='use CONV as graph convolutional layer for heterogenous edges (default: SAGE)')
parser.add_argument('--homo-edge-convolution', choices=homo_edge_conv_map.keys(), default='SAGE',
		    help='use CONV as graph convolutional layer for homogenous edges (default: SAGE)')
parser.add_argument('--learning-rate', type=float, default=0.001,
		    help='learning rate for the Adam optimizer')
parser.add_argument('--save-last-epoch-embedding', type=str, metavar='FILE',
		    help='save embeddings from last epoch to FILE')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = torch.load(args.input)
data = dataset[0].to(device)

# Add user node features:
data['author'].x = torch.eye(data['author'].num_nodes, device=device)
del data['author'].num_nodes

# Make graph undirected and remove reverse edge labels:
data = T.ToUndirected()(data)
del data['hotel', 'rev_ratings', 'author'].edge_label

def idx_invert_map(nodes, idx):
	'''In kfold_random_node_split() we split nodes into training and test sets.
	The nodes get reindexed, and we need to remap to new indexes in edge descriptions.
	This funciton creates the mapping for new indexes.
	'''
	lidx = idx.tolist()
	return torch.Tensor([(lidx.index(i) if i in lidx else -1) for i in range(nodes)]).to(torch.long).to(device=idx.device)

def kfold_random_node_split(k, data, node_type):
	'Create @k k-fold train-test splits of nodes of type @node_type from @data for k-fold cross validation'

	from copy import copy

	n_nodes = data[node_type].x.size(0)

	# create random permutation
	perm = torch.randperm(n_nodes, device=data[node_type].x.device)

	def _split(dat, idx):
		dat[node_type].x = dat[node_type].x[idx,:]
		dat[node_type].y = dat[node_type].y[idx,:]

		# prepare remapping tensor for edge indexes
		idx_map = idx_invert_map(n_nodes, idx)

		# keep only edges for existing nodes
		for edge_type in dat.metadata()[1]:
			if edge_type[0] == node_type and edge_type[2] == node_type:
				ends = [0, 1]
			elif edge_type[0] == node_type:
				ends = 0
			elif edge_type[2] == node_type:
				ends = 1
			else:
				continue

			# compute mask
			mask = torch.isin(dat[edge_type].edge_index[ends], idx)
			if mask.shape[0] == 2:
				mask = torch.logical_and(mask[0], mask[1])

			# drop edges for dropped nodes
			edge_index = dat[edge_type].edge_index.transpose(0, 1)[mask].transpose(0, 1)
	
			# remap node indexes in edges
			if ends == 0 or ends == [0, 1]:
				edge_index[0] = idx_map[edge_index[0]]
			if ends == 1 or ends == [0, 1]:
				edge_index[1] = idx_map[edge_index[1]]
				
			dat[edge_type].edge_index = edge_index

			# drop also labels according to mask
			if hasattr(dat[edge_type], 'edge_label'):
				dat[edge_type].edge_label = dat[edge_type].edge_label[mask]

	# generate the @k folds
	for i in range(k):
		lim = i * n_nodes // k, (i + 1) * n_nodes // k
		train_idx = torch.cat([perm[:lim[0]], perm[lim[1]:]])
		test_idx = perm[lim[0]:lim[1]]

		train_data, test_data = copy(data), copy(data)

		_split(train_data, train_idx)
		_split(test_data, test_idx)

		yield i, train_data, test_data

def train():
	model.train()
	optimizer.zero_grad()
	pred, _ = model(train_data.x_dict, train_data.edge_index_dict)
	target = train_data['hotel'].y
	loss = F.mse_loss(pred, target)
	loss.backward()
	optimizer.step()
	return float(loss)

@torch.no_grad()
def test(data):
	model.eval()
	pred, embedded = model(data.x_dict, data.edge_index_dict)
	target = data['hotel'].y.float()
	rmse = F.mse_loss(pred, target).sqrt()
	return float(rmse), (target, pred, embedded)

def trainable_parameter_count(model):
	model_parameters = filter(lambda p: p.requires_grad, model.parameters())
	return sum([np.prod(p.size()) for p in model_parameters])

epochs = args.epochs
folds = args.k_folds
confidence_interval = 0.99

record = {'times': []}
for key in 'loss', 'train_rmse', 'test_rmse':
	record[key] = [[] for i in range(epochs)]

print(f'Training on {torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU"}')
trainable_parameters_printed = False

for fold, train_data, test_data in kfold_random_node_split(folds, data, 'hotel'):
	model = NodeLabelPredModel('hotel',
				   nlayers=args.layers,
				   hidden_channels=args.hidden_channels,
				   out_channels=1,
				   metadata=data.metadata(),
				   hetero_edge_conv=args.hetero_edge_convolution,
				   homo_edge_conv=args.homo_edge_convolution
				  ).to(device)

	# Run one model step so the number of parameters can be inferred:
	with torch.no_grad():
		model(train_data.x_dict, train_data.edge_index_dict)

	if not trainable_parameters_printed:
		print(f'\nNumber of trainable model parameters: {trainable_parameter_count(model)}\n')
		trainable_parameters_printed = True

	optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

	beg = time()

	for epoch in range(epochs):
		loss = train()
		train_rmse, train_embedding = test(train_data)
		test_rmse, test_embedding = test(test_data)

		# save embeddings for visualization with visualization.py
		if args.save_last_epoch_embedding and epoch == epochs - 1 and fold == 0:
			print(f'saving last epoch embeddings of first fold to file {args.save_last_epoch_embedding}')
			embeddings = train_embedding, test_embedding
			torch.save(embeddings, args.save_last_epoch_embedding)

		record['loss'][epoch].append(loss)
		record['train_rmse'][epoch].append(train_rmse)
		record['test_rmse'][epoch].append(test_rmse)

		print(f'Fold: {fold+1:02d}, Epoch: {epoch+1:03d}, Loss: {loss:.4f}, Train: {train_rmse:.4f}, Test: {test_rmse:.4f}', flush=True)

	end = time()
	record['times'].append(end - beg)
	print(f'elapsed time: {end-beg:.06f} seconds')

double_sided_quantil = 1.0 - (1.0 - confidence_interval) / 2.0
conf_int_mul = scipy.stats.t.ppf(double_sided_quantil, df=folds - 1)

print(f'\nMean training time: {sum(record["times"])/folds:.06f} seconds')
print(f'Losses with {confidence_interval*100}% confidence intervals\n')

for epoch in range(epochs):
	print(f'Epoch: {epoch+1:03d}', end='')
	for key in 'loss', 'train_rmse', 'test_rmse':
		vals = record[key][epoch]
		mean = sum(vals) / folds
		mean_std = (sum([(val - mean)**2 for val in vals]) / (folds * (folds-1)))**0.5
		lo = mean - conf_int_mul * mean_std
		hi = mean + conf_int_mul * mean_std
		print(f' {key} {lo:.4f} {mean:.4f} {hi:.4f}', end='')
	print('')
