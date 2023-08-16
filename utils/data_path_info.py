import os

class DataPathInfo:
	'A simple class to represent the data file paths'

	def __init__(self, dir):
		self.set_dir(dir)

	def set_dir(self, dir):
		self.dir = dir
		self.authors_json_path = os.path.join(dir, 'authors.json')
		self.hotels_json_path = os.path.join(dir, 'hotels.json')
		self.reviews_json_path = os.path.join(dir, 'reviews.json')

	def mkdir(self):
		if not os.path.isdir(self.dir):
			os.mkdir(self.dir)
