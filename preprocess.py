#!/usr/bin/env python3

import argparse
import json
import os.path
import itertools
from tqdm import tqdm
from utils.data_path_info import DataPathInfo

INPUT = DataPathInfo('scraped')
OUTPUT = DataPathInfo('preprocessed')

def load_review_ids():
	"Load review, author and hotel IDs"

	reviews = {}

	# load review id, author id and offering id
	for line in tqdm(iterable=open(INPUT.reviews_json_path, 'r'),
			 desc=f'Loading review IDs from file {INPUT.reviews_json_path}'):
		d = json.loads(line)

		a_id = d['author_id']
		o_id = d['hotel_id']
		r_id = d['id']

		reviews[r_id] = (a_id, o_id)

	load_review_ids.cnt = len(reviews)
	print(f'Loaded {load_review_ids.cnt} review IDs')

	return reviews

def count_by_author_and_hotel(reviews):
	"Count number of reviews for each author and each hotel"

	by_author = {}
	by_hotel = {}

	for r_id, (a_id, o_id) in tqdm(iterable=reviews.items(), total=len(reviews), desc='Counting by author and hotel'):
		if a_id in by_author:
			by_author[a_id] += 1
		else:
			by_author[a_id] = 1

		if o_id in by_hotel:
			by_hotel[o_id] += 1
		else:
			by_hotel[o_id] = 1

	return by_author, by_hotel

def filter_by_min_reviews_by_author_hotel(reviews, min_author_reviews=1, min_hotel_reviews=1):
	"""Filter away those reviews which were authored by author who created
	fewer than given number of reviews, and those reviews which review
	hotel that has fewer than given number of reviews."""

	by_author, by_hotel = count_by_author_and_hotel(reviews)

	# filter author IDs with at least min_author_reviews reviews
	a_ids = { a_id
		  for a_id, num in by_author.items()
		  if num >= min_author_reviews }

	# filter hotel IDs with at least min_hotel_reviews reviews
	o_ids = { o_id
		  for o_id, num in by_hotel.items()
		  if num >= min_hotel_reviews }

	# return only review IDs which have author ID in a_ids and hotel ID in o_ids
	return { r_id: (a_id, o_id)
		 for r_id, (a_id, o_id) in reviews.items()
		 if a_id in a_ids and o_id in o_ids }

def load_and_filter_review_ids(min_author_reviews=1, min_hotel_reviews=1):
	"""Load review IDs and iteratively filter away reviews so that the result
	contains only reviews
	- authored by users that have at least given number of reviews in the result
	- for hotels that have at least given number of reviews in the result"""

	# load valid review IDs
	reviews = load_review_ids()

	old_cnt = len(reviews)
	rnd = 1
	diff = 1

	# iteratively filter away reviews so that the resulting list satisfies
	# the minimum author links and minimum hotel links condition
	while diff > 0:
		reviews = filter_by_min_reviews_by_author_hotel(reviews, min_author_reviews, min_hotel_reviews)

		cnt = len(reviews)
		diff = old_cnt - cnt
		old_cnt = cnt

		print(f'deleted {diff} reviews in round {rnd}, left with {cnt} reviews')

		rnd += 1

	return { r_id for r_id in reviews }

def preprocess(review_ids_to_keep, popular_amenities=15, popular_languages=5):
	wr = open(OUTPUT.reviews_json_path, 'w')
	authors = set()
	hotels = set()

	pbar = tqdm(total=load_review_ids.cnt, desc='Preprocessing reviews')
	for line in open(INPUT.reviews_json_path, 'r'):
		d = json.loads(line)
		pbar.update(1)

		# skip reviews not in review_ids_to_keep
		if d['id'] not in review_ids_to_keep:
			continue

		a_id = d['author_id']
		o_id = d['hotel_id']

		# put overall rating to d['rating']
		d['rating'] = d['ratings']['Overall']
		del d['ratings']['Overall']

		# put other ratings to d['rating_' + type]
		for key in 'Sleep Quality', 'Cleanliness', 'Location', 'Rooms', 'Service', 'Value':
			if key not in d['ratings']:
				continue

			new_key = 'rating_' + key.lower().replace(' ', '_')
			d[new_key] = d['ratings'][key]

		del d['ratings']

		# add author record if not already done
		if a_id not in authors:
			authors.add(a_id)

		# add hotel record if not already done
		if o_id not in hotels:
			hotels.add(o_id)

		# drop textual info
		del d['title'], d['text'], d['room_tip']

		wr.write(json.dumps(d) + '\n')
	wr.close()
	pbar.close()

	authors_written = set() # in case of duplicities
	wa = open(OUTPUT.authors_json_path, 'w')
	for line in tqdm(iterable=open(INPUT.authors_json_path, 'r'), desc='Preprocessing authors'):
		a = json.loads(line)

		# skip author with no reviews after preprocessing
		if a['id'] not in authors or a['id'] in authors_written:
			continue

		wa.write(json.dumps(a) + '\n')
		authors_written.add(a['id'])
	wa.close()
	del authors_written

	amenity_counts = {}
	lang_counts = {}
	for line in tqdm(iterable=open(INPUT.hotels_json_path, 'r'), desc='Counting hotel amenities and languages'):
		o = json.loads(line)

		# skip hotels with no reviews after preprocessing
		if o['id'] not in hotels:
			continue

		for amenity in o['amenities']:
			if amenity in amenity_counts:
				amenity_counts[amenity] += 1
			else:
				amenity_counts[amenity] = 1

		for lang in o['languages']:
			if lang in lang_counts:
				lang_counts[lang] += 1
			else:
				lang_counts[lang] = 1

	
	most_popular_amenities = { k for k, v in itertools.islice(sorted(amenity_counts.items(), key=lambda i: i[1], reverse=True), popular_amenities) }
	most_popular_languages = { k for k, v in itertools.islice(sorted(lang_counts.items(), key=lambda i: i[1], reverse=True), popular_languages) }

	wo = open(OUTPUT.hotels_json_path, 'w')
	for line in tqdm(iterable=open(INPUT.hotels_json_path, 'r'), desc='Preprocessing hotels'):
		o = json.loads(line)

		# skip hotels with no reviews after preprocessing
		if o['id'] not in hotels:
			continue

		del o['url'], o['name'], o['email'], o['phone'], o['address'], o['description']
		o['amenities'] = [ amenity for amenity in o['amenities'] if amenity in most_popular_amenities ]
		o['languages'] = [ lang for lang in o['languages'] if lang in most_popular_languages ]

		wo.write(json.dumps(o) + '\n')
	wo.close()

	print(f'{len(review_ids_to_keep)} reviews of {len(hotels)} hotels from {len(authors)} authors')

parser = argparse.ArgumentParser(description='Utility to preprocess scraped TripAdvisor dataset')
parser.add_argument('-A', '--min-author-reviews', type=int, default=1, metavar='N',
		    help='each author in the result must have at least N reviews after preprocessing')
parser.add_argument('-H', '--min-hotel-reviews', type=int, default=1, metavar='N',
		    help='each hotel in the result must have at least N reviews after preprocessing')
parser.add_argument('--amenities', type=int, default=15, metavar='N',
		    help='keep N most popular hotel amenities (default: 15)')
parser.add_argument('--languages', type=int, default=5, metavar='N',
		    help='keep N most popular hotel languages (default: 5)')
parser.add_argument('-i', '--input-dir', type=str, default='scraped',
		    help='input directory containing scraped data in json format (default: ./scraped)')
parser.add_argument('-o', '--output-dir', type=str, default='preprocessed',
		    help='output directory to write preprocessed json files (default: ./preprocessed)')

def main():
	args = parser.parse_args()

	if args.input_dir is not None:
		print(f'setting input dir to {args.input_dir}')
		INPUT.set_dir(args.input_dir)

	if args.output_dir is not None:
		print(f'setting output dir to {args.output_dir}')
		OUTPUT.set_dir(args.output_dir)
	OUTPUT.mkdir()

	review_ids_to_keep = load_and_filter_review_ids(args.min_author_reviews, args.min_hotel_reviews)
	preprocess(review_ids_to_keep, popular_amenities=args.amenities, popular_languages=args.languages)

if __name__ == '__main__':
	main()
