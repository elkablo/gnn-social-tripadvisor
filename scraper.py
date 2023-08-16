#!/usr/bin/env python3

import os
import re
import html
import json
import requests
import argparse
from utils.data_path_info import DataPathInfo

OUTPUT = DataPathInfo('scraped')

class JSONSetEncoder(json.JSONEncoder):
	'JSON Encoder extension that supports encoding python sets'
	def default(self, obj):
		if isinstance(obj, set):
			return list(obj)
		else:
			return json.JSONEncoder.default(self, obj)

urqlcache_re = re.compile('"urqlCache":(.*),"redux":')

def parse_urqlcache(contents):
	'''The reviews are stored in a json object in the webpage called urqlCache.
	Find this object and return it as a dictionary.'''
	urqlcache = urqlcache_re.search(contents)[1]

	x = json.loads(urqlcache)
	for k in x.keys():
		x[k] = json.loads(x[k]['data'])

	return x

def parse_hotel_page(contents):
	'''Parse hotel info from hotel page contents.
	The infromation is found in the urqlCache JSON object.'''

	urqlcache = parse_urqlcache(contents)

	numRooms = None
	contactLinks = []

	# Because each page may return the urqlcache object with slightly
	# different keys, we apply some heuristics to find the needed
	# information.
	for k in urqlcache:
		if not contactLinks:
			try:
				contactLinks = urqlcache[k]['currentLocation'][0]['businessAdvantageData']['contactLinks']
				if contactLinks is None:
					contactLinks = []
			except:
				pass

		if 'locations' not in urqlcache[k]:
			continue

		l = urqlcache[k]['locations'][0]
		if l is None:
			continue
		if 'locationDescription' in l:
			hotelInfo = l
		elif 'localizedStreetAddress' in l:
			address = l['localizedStreetAddress']['fullAddress']
			try:
				numRooms = l['detail']['hotel']['details']['numRooms']
			except:
				numRooms = None
		elif 'reviewListPage' in l:
			reviewListPage = l['reviewListPage']
		elif 'url' in l:
			url = l['url']
			latitude = l['latitude']
			longitude = l['longitude']

	# store the found info
	hotelInfo['fullAddress'] = address
	hotelInfo['contactLinks'] = contactLinks
	hotelInfo['numRooms'] = numRooms
	hotelInfo['url'] = url
	hotelInfo['gps'] = latitude, longitude

	return hotelInfo, reviewListPage

class Author:
	'A class that represents review authors'

	# static variables: author database and set of authors that need to be saved
	AUTHOR_DB = {}
	NONSAVED_AUTHORS = set()

	def __new__(cls, profile):
		# if the author is already in our database, do not create a new object
		id = profile['userId']
		if id in Author.AUTHOR_DB:
			return Author.AUTHOR_DB[id]
		else:
			return object.__new__(cls)

	def __init__(self, profile):
		# skip if author info already filled out
		if 'id' in Author.AUTHOR_DB:
			return

		self.id = profile['userId']
		self.username = profile['username']
		self.display_name = profile['displayName']
		if profile['hometown']['location'] is not None:
			self.hometown = profile['hometown']['location']['additionalNames']['long']
			self.hometown_id = profile['hometown']['locationId']
		else:
			self.hometown = self.hometown_id = None

		Author.AUTHOR_DB[self.id] = self
		Author.NONSAVED_AUTHORS.add(self.id)

	def to_json(self):
		v = { key: val for key, val in vars(self).items() if key[0] != '_' }
		return json.dumps(v)

	def save(self, f):
		print(self.to_json(), file=f)

	@staticmethod
	def save_all(f):
		for id in Author.NONSAVED_AUTHORS:
			Author.AUTHOR_DB[id].save(f)
		f.flush()
		Author.NONSAVED_AUTHORS.clear()

	@staticmethod
	def load_saved():
		print(f'Loading saved authors from {OUTPUT.authors_json_path}')
		for line in open(OUTPUT.authors_json_path, 'r'):
			author = object.__new__(Author)
			author.__dict__ = json.loads(line)
			Author.AUTHOR_DB[author.id] = author

class Review:
	'A class that represents hotel reviews'

	def __init__(self, review, hotel):
		self.id = review['id']
		self._hotel = hotel
		self.hotel_id = hotel.id
		self.created = review['createdDate']
		self.published = review['publishedDate']
		self.title = review['title']
		self.text = review['text']
		self.room_tip = review['roomTip']
		if review['tripInfo'] is not None:
			self.stay_date = review['tripInfo']['stayDate']
			self.trip_type = review['tripInfo']['tripType']
			if self.trip_type == 'NONE':
				self.trip_type = None
		else:
			self.stay_date = self.trip_type = None

		self.ratings = { 'Overall': review['rating'] }
		for rating in review['additionalRatings']:
			self.ratings[rating['ratingLabel']] = rating['rating']

		self._author = Author(review['userProfile'])
		self.author_id = self._author.id

	def to_json(self):
		v = { key: val for key, val in vars(self).items() if key[0] != '_' }
		return json.dumps(v)

	def save(self, f):
		print(self.to_json(), file=f)

class Hotel:
	'A class that represents hotels'
	HOTEL_IDS = set()

	def __init__(self, hotelInfo, reviewListPage):
		self.id = hotelInfo['locationId']
		self.region_id = hotelInfo['parent']['locationId']
		self.url = hotelInfo['url']

		page_url_prefix = '/Hotel_Review-g%d-d%d-Reviews-' % (self.region_id, self.id)
		page_url_suffix = self.url[len(page_url_prefix) - 1:]
		self._page_url_f = 'https://www.tripadvisor.com' + page_url_prefix + 'or%d' + page_url_suffix

		self.gps = hotelInfo['gps']
		self.name = hotelInfo['name']
		self.type = hotelInfo['accommodationType']
		self.description = hotelInfo['locationDescription']
		self.category = hotelInfo['accommodationCategory']

		amenities = hotelInfo['detail']['hotelAmenities']
		self.amenities = set()
		self.languages = set()

		if amenities is not None:
			for h in "highlightedAmenities", "nonHighlightedAmenities":
				if h not in amenities or amenities[h] is None:
					continue
				for t in amenities[h]:
					for amenity in amenities[h][t]:
						self.amenities.add(amenity['amenityNameLocalized'])

			if 'languagesSpoken' in amenities:
				self.languages = { lang['amenityNameLocalized'] for lang in amenities['languagesSpoken'] }
	
		self.stars = float(hotelInfo['detail']['starRating'][0]['tagNameLocalized'].split(' ')[0])
		self.address = hotelInfo['fullAddress']
		self.num_rooms = hotelInfo['numRooms']

		self.has_website = False
		self.phone = None
		self.email = None

		for contact in hotelInfo['contactLinks']:
			if contact['contactLinkType'] == 'URL_HOTEL':
				self.has_website = True
			elif contact['contactLinkType'] == 'PHONE':
				self.phone = contact['displayPhone']
			elif contact['contactLinkType'] == 'EMAIL':
				self.email = ''.join(contact['emailParts'])

		self._review_count = reviewListPage['totalCount']

		self._review_ids = set()
		self._first_page_reviews = self.parse_reviews(reviewListPage)

	def parse_reviews(self, reviewListPage):
		'''Parse reviews from the reviewListPage
		dictionary found in the urqlCache object'''
		result = {}

		for rev in reviewListPage['reviews']:
			# skip reviews with anonymous authors
			if rev['userProfile'] is None:
				continue

			review = Review(rev, self)
			result[review.id] = review

		return result

	def _save_reviews(self, f, reviews):
		'Save reviews to file f'
		for id, review in reviews.items():
			# skip reviews that are alredy saved
			if id in self._review_ids:
				continue

			self._review_ids.add(id)
			review.save(f)

	def iter_review_pages(self):
		'Iterate over pages of reviews'
		for i in range(10, self._review_count, 10):
			yield self._page_url_f % (i,)

	def load_saved_review_ids(self):
		'Loads review IDs of this hotel that we already have saved'
		for line in open(OUTPUT.reviews_json_path, 'r'):
			d = json.loads(line)
			if d['hotel_id'] == self.id:
				self._review_ids.add(d['id'])

	def get_and_save_reviews(self, f, f_authors, session):
		'Scrap and save reviews to file @f and their authors to file @f_authors'
		self._save_reviews(f, self._first_page_reviews)

		# to know how many pages to skip
		skip_review_count = len(self._review_ids) - len(self._first_page_reviews)

		for page_url in self.iter_review_pages():
			# should we skip downloading this page?
			if skip_review_count > 10:
				skip_review_count -= 10
				continue

			print(f' get reviews page {page_url}... ', end='', flush=True)
			req = session.get(page_url, timeout=30)
			print('done')
			contents = req.content.decode('utf-8')
			_, reviewListPage = parse_hotel_page(contents)
			reviews = self.parse_reviews(reviewListPage)
			self._save_reviews(f, reviews)
			Author.save_all(f_authors)

	def to_json(self):
		v = { key: val for key, val in vars(self).items() if key[0] != '_' }
		return json.dumps(v, cls=JSONSetEncoder)

	def save(self, f):
		# skip if already saved
		if self.id in Hotel.HOTEL_IDS:
			return

		print(f'Saving hotel {self.name} with {self._review_count} reviews')
		print(self.to_json(), file=f, flush=True)
		Hotel.HOTEL_IDS.add(self.id)

	@staticmethod
	def from_url(url, session):
		if url.startswith('http://'):
			url = 'https://' + url[7:]
		elif not url.startswith('https://'):
			url = 'https://www.tripadvisor.com/' + url

		print(f'get hotel page {url}... ', end='', flush=True)
		req = session.get(url, timeout=30)
		print('done')
		contents = req.content.decode('utf-8')

		return Hotel(*parse_hotel_page(contents))

	@staticmethod
	def load_saved_ids():
		print(f'Loading saved hotel IDs from {OUTPUT.hotels_json_path}')
		for line in open(OUTPUT.hotels_json_path, 'r'):
			Hotel.HOTEL_IDS.add(json.loads(line)['id'])

	@staticmethod
	def hotel_urls_by_region(region, session):
		'Scrap hotel URLs by region and return them by yielding'

		first_url = f'https://www.tripadvisor.com/Hotels-g{region}'
		page_url_fmt = f'https://www.tripadvisor.com/Hotels-g{region}-oa%d.html'

		# get first page
		req = session.get(first_url, timeout=30)
		contents = req.content.decode('utf-8')

		# find number of hotels in region
		search = re.search('<span class=\'highlight\'>([0-9]+) properties</span>', contents)
		if search is None:
			return
		nhotels = int(search[1])

		hotel_url_re = re.compile(f'"/Hotel_Review-g{region}-d[0-9]+-Reviews[^#"]+\.html')

		yielded_urls = set()

		# yield hotels from first page
		for url in set(hotel_url_re.findall(contents)):
			url = url[1:]
			if url in yielded_urls:
				continue
			yielded_urls.add(url)
			url = 'https://www.tripadvisor.com' + url
			yield url

		for page in range(30, nhotels, 30):
			# get page
			req = session.get(page_url_fmt % (page,), timeout=30)
			contents = req.content.decode('utf-8')

			# yield hotels from this page
			for url in set(hotel_url_re.findall(contents)):
				url = url[1:]
				if url in yielded_urls:
					continue
				yielded_urls.add(url)
				url = 'https://www.tripadvisor.com' + url
				yield url

parser = argparse.ArgumentParser(description='Utility to scrap hotel review data from TripAdvisor')
parser.add_argument('-r', '--region', type=int, required=True, action='append', metavar='REG',
		    help='scrap hotel from region REG; can be added multiple times')
parser.add_argument('-o', '--output-dir', type=str, default='scraped',
		    help='output directory to write scraped data in json format (default: ./scraped)')

def main():
	args = parser.parse_args()

	if args.output_dir is not None:
		print(f'setting output dir to {args.output_dir}')
		OUTPUT.set_dir(args.output_dir)
	OUTPUT.mkdir()

	# create http session
	session = requests.Session()
	session.headers['User-Agent'] = 'Wget/1.21.3'
	session.headers['Accept'] = '*/*'

	# open files
	hotels_fd = open(OUTPUT.hotels_json_path, 'a')
	reviews_fd = open(OUTPUT.reviews_json_path, 'a')
	authors_fd = open(OUTPUT.authors_json_path, 'a')

	# load saved hotels so that we do not download their reviews again
	Hotel.load_saved_ids()

	# load saved authors so that we do not duplicate them
	Author.load_saved()

	# for each region given in the command line
	for region in args.region:

		# get URLs for all hotels in that region
		for hotel_url in Hotel.hotel_urls_by_region(region, session):

			# get the first page of hotel reviews
			print(f'get hotel {hotel_url}... ', end='', flush=True)
			req = session.get(hotel_url, timeout=30)
			print('done')
			contents = req.content.decode('utf-8')

			# create hotel object
			hotel = Hotel(*parse_hotel_page(contents))

			# skip if hotel already saved
			if hotel.id in Hotel.HOTEL_IDS:
				print(f'Skipping hotel {hotel.name}')
				continue

			# get reviews and save
			hotel.load_saved_review_ids()
			hotel.get_and_save_reviews(reviews_fd, authors_fd, session)
			hotel.save(hotels_fd)

if __name__ == '__main__':
	main()
