#!/usr/bin/env python3

import networkx as nx
from networkx.algorithms.centrality import eigenvector_centrality
from networkx.algorithms.components import connected_components
import itertools
import argparse
from utils.cache import cache
from utils.data_reader import TripAdvisorDataReader
from utils.temporal import to_year_month, iter_months
from projection import remove_edges_with_low_n, remove_nodes_with_low_neighbors, get_hotels_with_at_least_reviews
from tqdm import tqdm
from copy import copy
from time import time

data_reader = TripAdvisorDataReader()

def get_reviews_by_date(filt=None):
	'Get a map from (year, month) tuples to lists of review tuples (hotel_id, author_id)'

	reviews_by_date = {}

	pbar = tqdm(total=data_reader.reviews_filesize(), desc='Mapping reviews by date')

	for review, pos in data_reader.reviews(filt, with_pos=True):
		date = to_year_month(review['published'])
		item = review['hotel_id'], review['author_id']
		if date in reviews_by_date:
			reviews_by_date[date].append(item)
		else:
			reviews_by_date[date] = [ item ]

		pbar.update(pos - pbar.n)

	pbar.close()

	return reviews_by_date

def create_temporal_hotel_projection(nreviews, min_common_authors, min_neighbors):
	'''Creates a bipartite network projection to hotels.
	Each edge represents a common association between hotels (how many
	common authors of reviews for the pair of hotels).
	Each edge contains this information for every month, i.e. each edge has
	these attributes with YYYY/MM keys:
	  'n_YYYYMM' - number of associations until YYYY/MM
	  'w_YYYYMM' - number of associations until YYYY/MM divided by max number
	               (per whole graph) of common associations until YYYY/MM
	'''

	print(f'Determining hotels with at least {nreviews} reviews')
	# consider only hotels that have at least @nreviews overall
	hotels = get_hotels_with_at_least_reviews(nreviews)

	# get reviews for hotels, by publishing date
	reviews = get_reviews_by_date(lambda d: d['hotel_id'] in hotels)

	min_date = min(reviews.keys())
	max_date = max(reviews.keys())
	print(f'Date range of reviews: {min_date[0]}/{min_date[1]:02d} - {max_date[0]}/{max_date[1]:02d}')

	# this will map authors to sets of hotels,
	# and be updated by adding each new month of review
	aid_hotels = {}

	print('Creating temporal graph')

	# our temporal graph
	G = nx.Graph()

	# here we store number of non-zero edges by date
	n_edges_by_date = {}

	# dictionary of zero n and w attributes until current month, to be
	# used in initialization of an edge
	zeros = {}

	prev_n_key = None
	first_date = None
	for date in iter_months(min_date, max_date):
		# edge attribute key for number of common associations (n)
		# and weight (number of common associations divided by max
		# number of common associations)
		date_key = f'{date[0]}{date[1]:02d}'
		n_key = f'n_{date_key}'
		w_key = f'w_{date_key}'
		eigen_centr_key = f'ec_{date_key}'

		def incr_edge(u, v):
			if not G.has_edge(u, v):
				G.add_edge(u, v, **zeros)
			if n_key in G[u][v]:
				G[u][v][n_key] += 1
			else:
				G[u][v][n_key] = 1

		if True:
			# group hotels by author for current month
			aid_hotels_cur = {}
			if date not in reviews:
				reviews[date] = []
			for aid, hotels_it in itertools.groupby(reviews[date], lambda r: r[1]):
				hotels = { h[0] for h in hotels_it }
				aid_hotels_cur[aid] = hotels

			# first consider the pairs of hotels of each author only from the current month
			for aid, hotels in aid_hotels_cur.items():
				for u, v in itertools.combinations(hotels, 2):
					incr_edge(u, v)

			# afterwards consider for each author the pair of hotels where first hotel
			# is from current month and second hotel is from up until current month
			for aid in set(aid_hotels_cur.keys()).intersection(aid_hotels.keys()):
				hotels_prev = aid_hotels[aid]
				hotels_cur = aid_hotels_cur[aid] - hotels_prev

				for u, v in itertools.product(hotels_cur, hotels_prev):
					incr_edge(u, v)

			# finally add author's hotels from current month to all-time map
			for aid, hotels in aid_hotels_cur.items():
				if aid in aid_hotels:
					aid_hotels[aid].update(hotels)
				else:
					aid_hotels[aid] = hotels
		else:
			# group hotels by author
			for aid, hotels_it in itertools.groupby(reviews[date], lambda r: r[1]):
				hotels = { h[0] for h in hotels_it }
				if aid in aid_hotels:
					aid_hotels[aid].update(hotels)
				else:
					aid_hotels[aid] = hotels

			for aid, hotels in aid_hotels.items():
				# for every two hotels u, v of given author increase the edge
				for u, v in itertools.combinations(hotels, 2):
					incr_edge(u, v)

		if prev_n_key is not None:
			# edge[n_key] now contains number of common associations
			# from this month only. Add also number of common associations
			# from previous months
			for u, v in G.edges:
				if n_key in G[u][v]:
					G[u][v][n_key] += G[u][v][prev_n_key]
				else:
					G[u][v][n_key] = G[u][v][prev_n_key]

		# Set edge weights for current date by dividing number of common associations
		# by the max number of common associations.
		max_n = max((G[u][v][n_key] for u, v in G.edges), default=1)
		for u, v in G.edges:
			G[u][v][w_key] = G[u][v][n_key] / max_n

		if G.number_of_edges() == 0:
			continue
		elif first_date is None:
			first_date = date

		# store number of non-zero edges for this date
		n_edges_by_date[date] = G.number_of_edges()

		beg = time()
		centralities = eigenvector_centrality(G, max_iter=1000, weight=w_key)
		end = time()

		nx.set_node_attributes(G, centralities, eigen_centr_key)

		print(f'{date[0]}/{date[1]:02d}: {len(reviews[date])} reviews, ' +\
		      f'graph has now {G.number_of_nodes()} nodes ' +\
		      f'and {G.number_of_edges()} edges, ' +\
		      f'eigenvector centrality computed in {end-beg:.02f}s')

		zeros[n_key] = 0
		zeros[w_key] = 0
		prev_n_key = n_key

	# remove edges that represent too few common authors at the end
	remove_edges_with_low_n(G, min_common_authors, n_key)

	# remove nodes which have too few neighbors
	remove_nodes_with_low_neighbors(G, min_neighbors)

	# save attributes
	G.graph['nodes'] = 'hotels'
	G.graph['min_reviews'] = nreviews
	G.graph['min_common'] = min_common_authors
	G.graph['min_neighbors'] = min_neighbors
	G.graph['min_date'] = first_date
	G.graph['max_date'] = max_date
	G.graph['n_edges_by_date'] = n_edges_by_date

	return G

def create_temporal_hotel_projection_(nreviews, min_common_authors, min_neighbors):
	G = nx.read_gpickle('hotel_temporal4.gpickle')

	for date in iter_months(G.graph['max_date'], G.graph['max_date']):
		w_key = f'w_{date[0]}{date[1]:02d}'

		b = time()
		centralities = eigenvector_centrality(G, weight=w_key)
		e = time()
		print(f'{w_key} {e-b:.06f}')
		print(centralities)

	exit(1)

parser = argparse.ArgumentParser(description='Utility to create TripAdvisor author or hotel temporal projection graph (Bipartite network projection with temporal edge attributes)')
parser.add_argument('-p', '--projection-to', choices=['authors', 'hotels'], default='authors',
		    help='create a graph of authors / hotels')
parser.add_argument('-r', '--min-reviews', type=int, default=20, metavar='N',
		    help='consider only authors/hotels with at least N reviews, default 20')
parser.add_argument('-c', '--min-common', type=int, default=3, metavar='N',
		    help='consider only authors who reviewed at least N common hotels with another author, ' +\
			 'or hotels with at least N common authors with another hotel, default 3')
parser.add_argument('-n', '--min-neighbors', type=int, default=3, metavar='N',
		    help='drop authors/hotels with less than N neighbors in the constructed graph, default 3')
parser.add_argument('-i', '--input-dir', type=str,
		    help='input directory containing scraped or processed data in json format')
parser.add_argument('-o', '--output', type=str, required=True,
		    help='file to output the graph to')

def main():
	global args, data_reader

	args = parser.parse_args()

	if args.input_dir:
		data_reader = TripAdvisorDataReader(args.input_dir)


	node_type = args.projection_to

	other_node_type = 'hotels' if node_type == 'authors' else 'authors'

	print(f'creating temporal {node_type} projection of {node_type} with')
	print(f'- each node having at least {args.min_reviews} reviews')
	print(f'- each node having at least {args.min_neighbors} neighbors at the last date')
	print(f'- each edge representing at least {args.min_common} common {other_node_type} at the last date')

	if node_type == 'authors':
		#G = create_author_projection(args.min_reviews, args.min_common, args.min_neighbors, args.max_year)
		print('not implemented')
		exit(1)
	else:
		G = create_temporal_hotel_projection(args.min_reviews, args.min_common, args.min_neighbors)

	nx.write_gpickle(G, args.output)

	print(f'saved to {args.output}')

if __name__ == '__main__':
	main()
