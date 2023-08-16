#!/usr/bin/env python3

import networkx as nx
from networkx.algorithms.components import connected_components
import itertools
import argparse
from utils.cache import cache
from utils.data_reader import TripAdvisorDataReader

data_reader = TripAdvisorDataReader()

@cache('n')
def get_authors_with_at_least_reviews(n):
	'Get the set of author IDs who created at least @n reviews'

	return set(aid for aid, cnt in data_reader.count_reviews_aggregated_by('author_id').items() if cnt >= n)

@cache('n')
def get_hotels_with_at_least_reviews(n):
	'Get the set of hotel IDs with at least @n reviews'

	return set(hid for hid, cnt in data_reader.count_reviews_aggregated_by('hotel_id').items() if cnt >= n)

def aggregate_reviews_field(aggr_field, field, values, max_year):
	'''Aggregate reviews field @field by field @aggr_field. Value of field @field must be
	present in set @values'''
	result = {}

	for d in data_reader.reviews_with_max_year_month(max_year):
		val = d[field]
		if val not in values:
			continue

		agg_val = d[aggr_field]
		if agg_val in result:
			result[agg_val].add(val)
		else:
			result[agg_val] = { val }

	return result

@cache('--min_reviews', 'max_year')
def get_hotel_review_authors_from(authors, max_year):
	'''Get a mapping of hotels to sets of author IDs: for each hotel, the set of author IDs
	who created reviews for that hotel. Only author IDs from the set @authors are considered.'''

	return aggregate_reviews_field('hotel_id', 'author_id', authors, max_year)

@cache('--min_reviews', 'max_year')
def get_author_review_hotels_from(hotels, max_year):
	'''Get a mapping of authors to sets of hotel IDs: for each author, the set of hotel IDs
	for which the author created reviews. Only hotel IDs from the set @hotels are considered.'''

	return aggregate_reviews_field('author_id', 'hotel_id', hotels, max_year)

@cache('nreviews', 'max_year')
def _create_author_projection(nreviews, max_year):
	authors = get_authors_with_at_least_reviews(nreviews)
	hotels = get_hotel_review_authors_from(authors, max_year)

	G = nx.Graph()
	# for every hotel get all authors that reviewed it
	for hid, authors in hotels.items():
		# and for every pair of authors
		for u, v in itertools.combinations(authors, 2):
			# increase association between the authors of that pair
			if G.has_edge(u, v):
				G[u][v]['n'] += 1
			else:
				G.add_edge(u, v, n=1)

	max_n = max(G[u][v]['n'] for u, v in G.edges)
	for u, v in G.edges:
		G[u][v]['w'] = G[u][v]['n'] / max_n

	return G

@cache('nreviews', 'max_year')
def _create_hotel_projection(nreviews, max_year):
	hotels = get_hotels_with_at_least_reviews(nreviews)
	authors = get_author_review_hotels_from(hotels, max_year)

	G = nx.Graph()
	# for every author get all hotels they reviewed
	for aid, hotels in authors.items():
		# and for every pair of hotels
		for u, v in itertools.combinations(hotels, 2):
			# increase association between the hotels of that pair
			if G.has_edge(u, v):
				G[u][v]['n'] += 1
			else:
				G.add_edge(u, v, n=1)

	max_n = max(G[u][v]['n'] for u, v in G.edges)
	for u, v in G.edges:
		G[u][v]['w'] = G[u][v]['n'] / max_n

	return G

def remove_edges_with_low_n(G, n, key='n'):
	'Remove projection edges where number of common associations is less than @n'

	# prepare a list of edges to remve
	rm = []
	for u, v in G.edges:
		if G[u][v][key] < n:
			rm.append((u, v))

	# remove them
	for u, v in rm:
		G.remove_edge(u, v)

	# remove isolated vertices
	G.remove_nodes_from(list(nx.isolates(G)))

	# it is possible that there are now multiple connected components,
	# some of them may have too few vertices. Prepare a set of nodes
	# from components with less than 10 vertices.
	rm = set()
	for c in connected_components(G):
		if len(c) < 10:
			rm.update(c)

	# remove those nodes
	G.remove_nodes_from(rm)

def remove_nodes_with_low_neighbors(G, n):
	'Remove nodes with number of neighbors less than @n'

	# repeat until all nodes have at least @n neighbors
	while True:
		rm = set()
		for u in G.nodes:
			if len(G[u]) < n:
				rm.add(u)
		if not rm:
			break
		G.remove_nodes_from(rm)

@cache('nreviews', 'min_common_hotels', 'min_neighbors', 'max_year')
def create_author_projection(nreviews, min_common_hotels, min_neighbors, max_year):
	G = _create_author_projection(nreviews, max_year)

	# remove edges that represent too few common hotels
	remove_edges_with_low_n(G, min_common_hotels)

	# remove nodes which have too few neighbors
	remove_nodes_with_low_neighbors(G, min_neighbors)

	# save attribtues
	G.graph['nodes'] = 'authors'
	G.graph['min_reviews'] = nreviews
	G.graph['min_common'] = min_common_hotels
	G.graph['min_neighbors'] = min_neighbors
	G.graph['max_year'] = max_year

	return G

@cache('nreviews', 'min_common_authors', 'min_neighbors', 'max_year')
def create_hotel_projection(nreviews, min_common_authors, min_neighbors, max_year):
	G = _create_hotel_projection(nreviews, max_year)

	# remove edges that represent too few common authors
	remove_edges_with_low_n(G, min_common_authors)

	# remove nodes which have too few neighbors
	remove_nodes_with_low_neighbors(G, min_neighbors)

	# save attributes
	G.graph['nodes'] = 'hotels'
	G.graph['min_reviews'] = nreviews
	G.graph['min_common'] = min_common_authors
	G.graph['min_neighbors'] = min_neighbors
	if max_year is not None:
		G.graph['max_year'] = max_year

	hotels_gps = { k: v['gps'] for k, v in data_reader.hotels_by_id('gps').items() }
	nx.set_node_attributes(G, hotels_gps, 'gps')

	return G

parser = argparse.ArgumentParser(description='Utility to create and analyze TripAdvisor authors or hotel projection graph (Bipartite network projection)')
parser.add_argument('--cache', action='store_true',
		    help='cache the various computations / use cached values')
parser.add_argument('-y', '--max-year', type=int, metavar='Y',
		    help='consider only reviews published at most in year Y')
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
		    help='input directory containing scraped or preprocessed data in json format')
parser.add_argument('-o', '--output', type=str, required=True,
		    help='file to output the graph to')

def main():
	global args, data_reader

	args = parser.parse_args()

	if args.input_dir:
		data_reader = TripAdvisorDataReader(args.input_dir)

	node_type = args.projection_to

	other_node_type = 'hotels' if node_type == 'authors' else 'authors'

	print(f'creating {other_node_type} projection to {node_type} with')
	print(f'- each node having at least {args.min_reviews} reviews')
	print(f'- each node having at least {args.min_neighbors} neighbors')
	print(f'- each edge representing at least {args.min_common} common {other_node_type}')
	if args.max_year is not None:
		print(f'- each review published at most in year {args.max_year}')

	if node_type == 'authors':
		G = create_author_projection(args.min_reviews, args.min_common, args.min_neighbors, args.max_year)
	else:
		G = create_hotel_projection(args.min_reviews, args.min_common, args.min_neighbors, args.max_year)

	print(f'the projection graph has {len(G.nodes)} nodes with {len(G.edges)} edges')

	nx.write_gpickle(G, args.output)

	print(f'saved to {args.output}')

if __name__ == '__main__':
	main()
