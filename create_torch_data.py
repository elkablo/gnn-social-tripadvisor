#!/usr/bin/env python3

import torch
from torch_geometric.data import HeteroData, InMemoryDataset
from sklearn.preprocessing import MultiLabelBinarizer
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal
import networkx as nx
import pandas as pd
import numpy as np
from utils.temporal import iter_months
from tqdm import tqdm
import argparse
import os.path

ALL_RATING_TYPES = ['rating', 'rating_value', 'rating_sleep_quality', 'rating_service', 'rating_rooms', 'rating_location', 'rating_cleanliness']

def to_torch(x):
	'Convert to torch tensor from pandas DataFrame, pandas Series, or numpy ndarray'
	if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
		return torch.from_numpy(x.values)
	elif isinstance(x, np.ndarray):
		return torch.from_numpy(x)
	else:
		return x

def convert_NaNs(x):
	'''Converts NaNs to zeros and for each column containing NaNs
	adds another binary column indicating whether a NaN was in the
	original data or not (0 - NaN, 1 - not NaN).
	'''
	x = to_torch(x)

	# first check whether there are any NaNs at all
	if x.isnan().sum() == 0:
		return x.to(torch.float)

	# indicate NaNs only for columns containing NaNs
	if len(x.shape) == 1:
		x_nans = x
	else:
		x_nans = x.transpose(0,1)[x.isnan().sum(0) > 0].transpose(0,1)

	# convert NaNs to zeros
	x_wo_nans = x.nan_to_num(0.0)

	# create NaN indicator tensor
	nan_indicator = (~x_nans.isnan()).to(torch.long)

	# concatenate values with NaN indicators
	if len(x.shape) == 1:
		return torch.stack([x_wo_nans, nan_indicator]).transpose(0, 1).to(torch.float)
	else:
		return torch.cat([x_wo_nans, nan_indicator], 1).to(torch.float)

def read_jsons(dir):
	'Read hotel, review and author json files'

	hotels = pd.read_json(os.path.join(dir, 'hotels.json'), lines=True)
	reviews = pd.read_json(os.path.join(dir, 'reviews.json'), lines=True)
	authors = pd.read_json(os.path.join(dir, 'authors.json'), lines=True)

	hotel_mapping = {idx: i for i, idx in enumerate(hotels['id'])}
	author_mapping = {idx: i for i, idx in enumerate(authors['id'])}

	return hotels, reviews, authors, hotel_mapping, author_mapping

def add_review_edges(data, reviews, hotel_mapping, author_mapping, rating_types):
	# create review edges between hotels and authors
	src = [author_mapping[idx] for idx in reviews['author_id']]
	dst = [hotel_mapping[idx] for idx in reviews['hotel_id']]
	edge_index = torch.tensor([src, dst])

	# add ratings as edge labels
	edge_labels = convert_NaNs(reviews[ALL_RATING_TYPES])

	data['author', 'ratings', 'hotel'].edge_index = edge_index
	data['author', 'ratings', 'hotel'].edge_label = edge_labels

def augment_for(data, graph_file, node_mapping, node_type):
	'''Add homogenous hotel-to-hotel or author-to-author edges from networkx @graph_file,
	also add centralities and communities, if they are present'''

	# read networkx graph
	G = nx.read_gpickle(graph_file)

	# add hotel/author communities if they are present
	if 'communities' in G.graph:
		print(f'adding {node_type} community information')
		_, ncommunities = G.graph['communities']

		if hasattr(data[node_type], 'x'):
			nnodes = data[node_type].x.shape[0]
		else:
			nnodes = data[node_type].num_nodes

		comtensor = torch.zeros(nnodes, ncommunities)

		# fill the new tensor with bit-encoded communities
		for gid, idx in node_mapping.items():
			if gid in G.nodes:
				comtensor[idx][G.nodes[gid]['community']] = 1

		# append to node.x
		if hasattr(data[node_type], 'x'):
			data[node_type].x = torch.cat([data[node_type].x, comtensor], dim=1)
		else:
			data[node_type].x = comtensor

	# add hotel/author centralities if they are present
	if 'centralities' in G.graph:
		print(f'adding {node_type} centralities information {G.graph["centralities"]}')

		if hasattr(data[node_type], 'x'):
			nnodes = data[node_type].x.shape[0]
		else:
			nnodes = data[node_type].num_nodes

		centensor = torch.zeros(nnodes, 1)

		# fill the new tensor with centralities
		for gid, idx in node_mapping.items():
			if gid in G.nodes:
				centensor[idx][0] = G.nodes[gid][G.graph['centralities']]

		# append to node.x
		if hasattr(data[node_type], 'x'):
			data[node_type].x = torch.cat([data[node_type].x, comtensor], dim=1)
		else:
			data[node_type].x = comtensor

	# create list
	src, dst = zip(*((node_mapping[u], node_mapping[v]) for u, v in G.edges if u in node_mapping and v in node_mapping))
	weights = [G[u][v]['w'] for u, v in G.edges if u in node_mapping and v in node_mapping]

	# because the edges are undirected
	src, dst = src + dst, dst + src
	weights = weights + weights

	# add edges and their weights to data
	data[node_type, 'to', node_type].edge_index = torch.tensor([src, dst])
	data[node_type, 'to', node_type].edge_label = torch.tensor(weights)

def augment(data, hotel_graph_file, hotel_mapping, author_graph_file, author_mapping):
	# add hotel-hotel edges from @hotel_graph_file
	if hotel_graph_file is not None:
		augment_for(data, hotel_graph_file, hotel_mapping, 'hotel')

	# add author-author edges from @author_graph_file
	if author_graph_file is not None:
		augment_for(data, author_graph_file, author_mapping, 'author')

def create_hotel_feature_tensor(hotels, with_hotel_class=True):
	# Hotel feautres are:
	# - has_website
	# - amenities
	# - languages
	# - hotel class (number of stars) (optionally)
	hotels_has_website = torch.from_numpy(hotels.has_website.values.reshape(len(hotels), 1))
	hotels_amenities = torch.from_numpy(MultiLabelBinarizer().fit_transform(hotels.amenities))
	hotels_langs = torch.from_numpy(MultiLabelBinarizer().fit_transform(hotels.languages))

	lst = [hotels_has_website, hotels_amenities, hotels_langs]

	if with_hotel_class:
		hotels_stars = torch.from_numpy(hotels.stars.values.reshape(len(hotels), 1))
		lst.append(hotels_stars)

	return torch.cat(lst, 1).to(torch.float)

def create_for_review_rating_prediction(dir, output, hotel_graph_file=None, author_graph_file=None):
	print('Creating torch data file for edge label (review rating) prediction')

	hotels, reviews, authors, hotel_mapping, author_mapping = read_jsons(dir)

	data = HeteroData()

	# Hotel feautres
	data['hotel'].x = create_hotel_feature_tensor(hotels)

	# No author features.
	#data['author'].x = convert_NaNs(authors[['num_cities', 'num_helpful_votes', 'num_reviews', 'num_type_reviews']])
	data['author'].num_nodes = len(author_mapping)

	# add review edges with overall rating (to be predicted)
	add_review_edges(data, reviews, hotel_mapping, author_mapping, 'rating')

	# add data from hotel projection and author projection
	augment(data, hotel_graph_file, hotel_mapping, author_graph_file, author_mapping)

	# save the dataset
	torch.save(InMemoryDataset.collate([data]), output)
	print(f'Dataset stored in {output}')

def create_for_hotel_class_prediction(dir, output, hotel_graph_file=None, author_graph_file=None):
	print('Creating torch data file for node class (hotel class) prediction')

	hotels, reviews, authors, hotel_mapping, author_mapping = read_jsons(dir)

	data = HeteroData()

	# Hotel feautres
	data['hotel'].x = create_hotel_feature_tensor(hotels, with_hotel_class=False)

	# Hotel class is:
	# - hotel class (number of stars)
	hotels_stars = torch.from_numpy(hotels.stars.values.reshape(len(hotels), 1))
	data['hotel'].y = hotels_stars.to(torch.float)

	# No author features.
	data['author'].num_nodes = len(author_mapping)

	# add review edges with all ratings
	add_review_edges(data, reviews, hotel_mapping, author_mapping, ALL_RATING_TYPES)

	# add data from hotel projection and author projection
	augment(data, hotel_graph_file, hotel_mapping, author_graph_file, author_mapping)

	# save the dataset
	torch.save(InMemoryDataset.collate([data]), output)
	print(f'Dataset stored in {output}')

def create_for_temporal_hotel_prediction(dir, output, temporal_hotels_graph_file):
	MIN_EDGES = 10000

	print('Creating torch data file for temporal edge weight prediction between hotels')

	# read networkx graph
	G = nx.read_gpickle(temporal_hotels_graph_file)

	# read hotel info
	hotels = pd.read_json(os.path.join(dir, 'hotels.json'), lines=True)

	# filter away hotels that are not in graph
	hotels = hotels[hotels['id'].isin(G.nodes)].reset_index(drop=True)

	# create mapping of hotel IDs to indexes, we need this for edge index tensor
	hotel_mapping = { idx: i for i, idx in enumerate(hotels['id']) }

	# fint first date when the graph has at least MIN_EDGES edges
	first_date = None
	for date in iter_months(G.graph['min_date'], G.graph['max_date']):
		if G.graph['n_edges_by_date'][date] >= MIN_EDGES:
			first_date = date
			break

	if first_date is None:
		print(f'At no date does the graph have at least {MIN_EDGES} edges')
		exit(1)

	# prepare hotel features, these are static but needs to be referenced in each
	# snapshot
	feature = create_hotel_feature_tensor(hotels)

	# prepare dynamic graph lists
	edge_indices = []
	edge_weights = []
	features = []
	targets = []

	for date in tqdm(list(iter_months(first_date, G.graph['max_date'])), desc='Appending months'):
		date_key = f'{date[0]}{date[1]:02d}'
		n_key = f'n_{date_key}'
		w_key = f'w_{date_key}'
		eigen_centr_key = f'ec_{date_key}'

		# create edges
		src, dst = zip(*((hotel_mapping[u], hotel_mapping[v]) for u, v in G.edges if G[u][v][n_key] > 0 and u in hotel_mapping and v in hotel_mapping))
		weights = [G[u][v][w_key] for u, v in G.edges if G[u][v][n_key] > 0]

		# because the edges are undirected
		src, dst = src + dst, dst + src
		weights = weights + weights

		# create edge tensors
		edge_index = torch.tensor([src, dst])
		edge_weight = torch.tensor(weights)

		# create target tensor
		centralities = [(G.nodes[idx][eigen_centr_key] if eigen_centr_key in G.nodes[idx] else 0.0) for idx in hotels['id']]
		target = torch.tensor(centralities, dtype=torch.float).reshape(len(hotels), 1)

		# add to list for temporal snapshots
		edge_indices.append(edge_index)
		edge_weights.append(edge_weight)
		features.append(feature)
		targets.append(target)

	# create dynamic temporal graph
	data = DynamicGraphTemporalSignal(edge_indices, edge_weights, features, targets)

	# save
	torch.save(data, output)
	print(f'Dataset stored in {output}')

parser = argparse.ArgumentParser(description='Utility to create torch data file from preprocessed TripAdvisor data')
parser.add_argument('-i', '--input-dir', type=str, default='preprocessed',
		    help='directory where to find preprocessed json files (default: ./preprocessed)')
parser.add_argument('--review-rating-prediction', action='store_true',
		    help='create hetero data for edge label (review rating) prediction')
parser.add_argument('--hotel-class-prediction', action='store_true',
		    help='create hetero data for node class (hotel class) prediction')
parser.add_argument('--augment-hotels-from', type=str, metavar='FILE',
		    help='add hotel centralities, communities, and edges between hotels from networkx graph file FILE (created with projection.py utility)')
parser.add_argument('--augment-authors-from', type=str, metavar='FILE',
		    help='add author centralities, communities, and edges between authors from networkx graph file FILE (created with projection.py utility)')
parser.add_argument('--temporal-hotel-prediction', type=str, metavar='FILE',
		    help='create dataset for temporal prediction of edges between hotels from networkx graph file FILE (create with temporal_projection.py utility)')
parser.add_argument('-o', '--output', type=str, required=True,
		    help='output file')

def main():
	args = parser.parse_args()

	if args.review_rating_prediction:
		create_for_review_rating_prediction(args.input_dir,
						    args.output, args.augment_hotels_from,
						    args.augment_authors_from)
	elif args.hotel_class_prediction:
		create_for_hotel_class_prediction(args.input_dir,
						  args.output, args.augment_hotels_from,
						  args.augment_authors_from)
	elif args.temporal_hotel_prediction:
		create_for_temporal_hotel_prediction(args.input_dir, args.output,
						     args.temporal_hotel_prediction)

if __name__ == '__main__':
	main()
