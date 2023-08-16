#!/usr/bin/env python3

import torch
from multiprocessing import cpu_count
import argparse
import numpy as np
from umap import UMAP
import matplotlib.pyplot as plt
import matplotlib

# set bigger font size
matplotlib.rcParams.update({'font.size': 22})

parser = argparse.ArgumentParser(description='GNN model embeddings visualizer')
parser.add_argument('-i', '--input', type=str, metavar='FILE', required=True,
		    help='input FILE containing the embeddings')
parser.add_argument('-p', '--prefix', type=str, required=True,
		    help='prefix for the output image file names')
parser.add_argument('-r', '--review-rating', action='store_true',
		    help='FILE contains embeddings for the review rating prediction task')
parser.add_argument('-H', '--hotel-class', action='store_true',
		    help='FILE contains embeddings for the hotel class prediction task')

def do_plot(outfile, data, color, s=25, alpha=0.5):
	print(f'plotting {outfile}')
	plt.figure(figsize=(30,15))
	plt.scatter(data[:,0], data[:,1], c=color, s=s, cmap='gist_rainbow', alpha=alpha, vmin=1, vmax=5)
	plt.colorbar()
	plt.axis('off')
	plt.savefig(outfile, pad_inches=0, bbox_inches='tight')

def visualize_link_label_embeddings(prefix, embeddings):
	# first unpack the tuples
	train, test = embeddings

	train_target, train_pred, (train_z1, train_z2) = train
	test_target, test_pred, (test_z1, test_z2) = test

	# prepare reducers
	z1reducer = UMAP(n_components=2, n_neighbors=200, n_jobs=cpu_count())
	z2reducer = UMAP(n_components=2, n_neighbors=200, n_jobs=cpu_count())

	# reduce z1
	print('reducing train z1')
	train_z1_low = z1reducer.fit_transform(train_z1)
	print('reducing test z1')
	test_z1_low = z1reducer.transform(test_z1)

	# reduce z2
	print('reducing train z2')
	train_z2_low = z2reducer.fit_transform(train_z2)
	print('reducing test z2')
	test_z2_low = z2reducer.transform(test_z2)

	# plot all files
	do_plot(f'{prefix}_train_z1_target.png', train_z1_low, train_target)
	do_plot(f'{prefix}_train_z1_predicted.png', train_z1_low, train_pred)
	do_plot(f'{prefix}_test_z1_target.png', test_z1_low, test_target)
	do_plot(f'{prefix}_test_z1_predicted.png', test_z1_low, test_pred)
	do_plot(f'{prefix}_train_z2_target.png', train_z2_low, train_target)
	do_plot(f'{prefix}_train_z2_predicted.png', train_z2_low, train_pred)
	do_plot(f'{prefix}_test_z2_target.png', test_z2_low, test_target)
	do_plot(f'{prefix}_test_z2_predicted.png', test_z2_low, test_pred)

def visualize_node_label_embeddings(prefix, embeddings):
	# first unpack the tuples
	train, test = embeddings

	train_target, train_pred, train_embs = train
	test_target, test_pred, test_embs = test

	def do_plot_node(out, data, color):
		do_plot(out, data, color, s=200, alpha=0.7)

	# visualize all layer embeddings
	for i in range(len(train_embs)):
		reducer = UMAP(n_components=2, n_neighbors=200, n_jobs=cpu_count())

		print(f'reducing layer {i+1}/{len(train_embs)}')
		print('  reducing train set')
		train_low = reducer.fit_transform(train_embs[i]['hotel'])
		print('  reducing test set')
		test_low = reducer.transform(test_embs[i]['hotel'])

		# plot
		do_plot_node(f'{prefix}_train_l{i+1}_target.png', train_low, train_target)
		do_plot_node(f'{prefix}_train_l{i+1}_predicted.png', train_low, train_pred)
		do_plot_node(f'{prefix}_test_l{i+1}_target.png', test_low, test_target)
		do_plot_node(f'{prefix}_test_l{i+1}_predicted.png', test_low, test_pred)

def main():
	args = parser.parse_args()

	embeddings = torch.load(args.input, map_location=torch.device('cpu'))

	train_embeddings, test_embeddings = embeddings

	if args.review_rating:
		visualize_link_label_embeddings(args.prefix, embeddings)
	elif args.hotel_class:
		visualize_node_label_embeddings(args.prefix, embeddings)
	else:
		parser.error('-l or -n is needed')

if __name__ == '__main__':
	main()
