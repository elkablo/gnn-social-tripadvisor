import json
import os

class TripAdvisorDataReader:
	'Reader class for scarped data from TripAdvisor'

	def __init__(self, data_dir=os.path.join(os.path.dirname(__file__), '../preprocessed')):
		self.authors_path = os.path.join(data_dir, 'authors.json')
		self.hotels_path = os.path.join(data_dir, 'hotels.json')
		self.reviews_path = os.path.join(data_dir, 'reviews.json')

		self.min_date = self.max_date = None

	def _get_min_max_date(self):
		'Finds minimal and maximal review publishing dates'

		min_date = max_date = None

		for line in open(self.reviews_path):
			d = json.loads(line)

			year = int(dct['published'][:4])
			month = int(dct['published'][5:7])
			day = int(dct['published'][8:10])

			date = year, month, day

			if min_date is None or date < min_date:
				min_date = date

			if max_date is None or date > max_date:
				max_date = date

		self.min_date, self.max_date = min_date, max_date

	def min_date(self):
		'Returns minimal review publish date'

		if self.min_date is None:
			self._get_min_max_date()

		return self.min_date

	def max_date(self):
		'Returns maximal review publish date'

		if self.max_date is None:
			self._get_min_max_date()

		return self.max_date

	def reviews_filesize(self):
		return os.stat(self.reviews_path).st_size

	def reviews(self, filt=None, with_pos=False):
		'Yields reviews, potentially filtered by @filt'

		pos = 0
		for line in open(self.reviews_path):
			pos += len(line)
			d = json.loads(line)

			# ignore reviews that don't satisfy filter
			if filt is not None and not filt(d):
				continue

			if with_pos:
				yield d, pos
			else:
				yield d

	def authors(self):
		'Yields authors'

		for line in open(self.authors_path):
			yield json.loads(line)

	def hotels(self):
		'Yields hotels'

		for line in open(self.hotels_path):
			yield json.loads(line)

	def hotels_by_id(self, fields=None):
		'Return map of hotels by ID, potentially only with fields @fields'

		hotels = {}

		for hotel in self.hotels():
			hid = hotel['id']
			if fields is not None:
				hotel = { k: v for k, v in hotel.items() if k in fields }
			hotels[hid] = hotel

		return hotels

	def count_reviews_aggregated_by(self, field, transform=None):
		'Count reviews aggregated by field @field, potentially tranformed by @tranform'

		counts = {}

		for d in self.reviews():
			value = d[field]

			# transform field value
			if transform is not None:
				value = transform(value)

			if value not in counts:
				counts[value] = 1
			else:
				counts[value] += 1

		return counts

	@staticmethod
	def review_filter_max_year_month(year, month=None):
		'Generate a review filter that filters away reviews with publishing date greater than @year and @month'

		if year is None:
			return None

		if month:
			def filt(dct):
				y = int(dct['published'][:4])
				m = int(dct['published'][5:7])
				return y < year or (y == year and m <= month)
		else:
			def filt(dct):
				return int(dct['published'][:4]) <= year

		return filt

	def reviews_with_max_year_month(self, year, month=None):
		'Yield reviews with publishing date at most @year and @month'

		return self.reviews(self.review_filter_max_year_month(year, month))
