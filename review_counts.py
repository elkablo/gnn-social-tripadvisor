#!/usr/bin/env python3

import argparse
from utils.data_reader import TripAdvisorDataReader

data_reader = TripAdvisorDataReader()

def count_reviews_yearly():
	'Count reviews by year'
	counts = data_reader.count_reviews_aggregated_by('created', lambda x: int(x[:4]))

	for year in range(min(counts.keys()), max(counts.keys()) + 1):
		if year not in counts:
			counts[year] = 0
		print(f'{year} {counts[year]}')

def iter_year_month(start, end):
	'Iterate year-month from @start to @end'

	cur = start
	while cur <= end:
		yield cur

		if cur[1] == 12:
			cur = cur[0] + 1, 1
		else:
			cur = cur[0], cur[1] + 1

def count_reviews_monthly():
	'Counts reviews by month'
	counts = data_reader.count_reviews_aggregated_by('created', lambda x: (int(x[:4]), int(x[5:7])))

	for year, month in iter_year_month(min(counts.keys()), max(counts.keys())):
		k = year, month
		if k not in counts:
			counts[k] = 0
		print(f'{year}-{month:02d} {counts[k]}')

def count_reviews_monthly_only():
	'Counts reviews by month only'
	counts = data_reader.count_reviews_aggregated_by('created', lambda x: int(x[5:7]))

	for month in range(1, 13):
		if month not in counts:
			counts[month] = 0
		print(f'{month:02d} {counts[month]}')

def count_active_authors_per_year(min_reviews=1):
	'Counts number of active authors per year'
	authors_by_year = {}

	for d in data_reader.reviews():
		year = int(d['created'][:4])
		aid = d['author_id']
		if year in authors_by_year:
			if aid in authors_by_year[year]:
				authors_by_year[year][aid] += 1
			else:
				authors_by_year[year][aid] = 1
		else:
			authors_by_year[year] = { aid: 1 }

	for year in range(min(authors_by_year.keys()), max(authors_by_year.keys()) + 1):
		n_reviews = sum(val for aid, val in authors_by_year[year].items() if val >= min_reviews)
		n_authors = sum(1 for aid, val in authors_by_year[year].items() if val >= min_reviews)
		avg = n_reviews / n_authors if n_authors > 0 else 0
		print(f'{year} {n_reviews} {n_authors} {avg:.04f}')

def hist_reviews_by_field(field, bin_width=1):
	'Dumps histogram of reviews by given field'
	from numpy import histogram

	counts = list(data_reader.count_reviews_aggregated_by(field).values())

	end = max(counts)
	modulo = end % bin_width
	if modulo:
		end += bin_width - modulo

	bins, ranges = histogram(counts, bins=end // bin_width, range=(1, end))
	for bin, val in enumerate(bins):
		print(f'{bin*bin_width} {val}')

parser = argparse.ArgumentParser(description='Analyze TripAdvisor reviews')
parser.add_argument('-y', '--yearly', action='store_true',
		    help='count reviews by year')
parser.add_argument('-m', '--monthly', action='store_true',
		    help='count reviews by year and month')
parser.add_argument('-M', '--monthly-only', action='store_true',
		    help='count reviews by month only')
parser.add_argument('--histogram-by-author', action='store_true',
		    help='count histogram of reviews by author')
parser.add_argument('--histogram-by-hotel', action='store_true',
		    help='count histogram of reviews by hotel')
parser.add_argument('-a', '--authors-per-year', action='store_true',
		    help='count active author per year')
parser.add_argument('--min-author-reviews', type=int, metavar='N', default=1,
		    help='consider only authors with at least N reviews  (default: 1)')
parser.add_argument('-i', '--input-dir', type=str,
		    help='input directory containing scraped or preprocessed data in json format')

def main():
	global args, data_reader

	args = parser.parse_args()

	if args.input_dir:
		data_reader = TripAdvisorDataReader(args.input_dir)

	if args.yearly:
		count_reviews_yearly()
	if args.monthly:
		count_reviews_monthly()
	if args.monthly_only:
		count_reviews_monthly_only()
	if args.histogram_by_author:
		hist_reviews_by_field('author_id')
	if args.histogram_by_hotel:
		hist_reviews_by_field('hotel_id', 100)
	if args.authors_per_year:
		count_active_authors_per_year(args.min_author_reviews)

if __name__ == '__main__':
	main()
