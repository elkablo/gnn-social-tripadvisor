#!/usr/bin/env python3
#
# Utility for plotting MNIST hand-written digits dataset via various dimensionality reduction methods

import argparse
from multiprocessing import cpu_count
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, SpectralEmbedding
import scipy.stats
from time import process_time
import pickle

# don't fail if UMAP is not installed
try:
	from umap import UMAP
except:
	UMAP = None

def plot_2d(data, fname):
	plt.figure(figsize=(30, 20))

	# scatter plot for each digit
	for digit in range(10):
		plt.scatter(data[digit][:, 0],
			    data[digit][:, 1],
			    s=20, label=str(digit), alpha=0.75)

	plt.legend()
	plt.savefig(fname)

def plot_3d(data, fname):
	fig3d = plt.figure(figsize=(12, 12))
	ax = fig3d.add_subplot(111, projection='3d')

	# rotate view for each frame
	def rot(n, ax):
		ax.view_init(30, n)

	# scatter plot for each digit
	for digit in range(10):
		ax.scatter(data[digit][:, 0],
			   data[digit][:, 1],
			   data[digit][:, 2],
			   s=3, label=str(digit), alpha=0.75)

	# create the animation and save as video
	animation.FuncAnimation(fig3d, rot, frames=360, interval=40, fargs=(ax,)).save(fname)

class MNIST_DimRed:
	'Base class for dimensionality reduction plotting on MNIST dataset'
	kwargs = {}

	def __init__(self, **kwargs):
		for key in kwargs.keys():
			if key not in self.kwargs:
				raise Exception(f'setting {key} not allowed for {self.reduction.__name__}')

		self.kwargs.update(kwargs)

	def get_cache_fname(self, samples, dims):
		'Get filename of cached transformation'

		kwargs_str = '_'.join([f'{k}={v}' for k, v in self.kwargs.items() if k != 'n_jobs'])
		if kwargs_str:
			kwargs_str = '_' + kwargs_str
	
		return f'.cache_{self.reduction.__name__}_{samples}_dims={dims}{kwargs_str}.bin'

	def _measure_performance_kfold(self, X, k):
		x = []
		from sys import stderr
		for i in range(k):
			beg = process_time()
			self.reduction(n_components=2, **self.kwargs).fit_transform(X)
			end = process_time()
			x.append(end - beg)
			print('.', flush=True, file=stderr, end='')
		mean = sum(x) / k
		sigma = (sum((v - mean)**2 for v in x) / (k*(k-1)))**0.5
		return mean, scipy.stats.t.ppf(0.975, df=k-1)*sigma

	def measure_time_performance(self, X):
		print('measuring performance', flush=True)

		for n in 1024, 2048, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 49152:
			mean, plusminus = self._measure_performance_kfold(X[:n, :], 10)
			print(f'{n} samples: {mean:.06f} secs', flush=True)
			#print(f'{n} samples: {mean:.06f} ± {plusminus:.06f} secs', flush=True)
			continue
			beg = process_time()
			self.reduction(n_components=2, **self.kwargs).fit_transform(X[:n, :])
			end = process_time()
			print(f'{n} samples: {end-beg:.06f} secs', flush=True)

	def plot(self, X, y, fname, video=False):
		'Do the dimensionality reduction transformation and plot the result'

		dims = 3 if video else 2
		cache = self.get_cache_fname(X.shape[0], dims)

		try:
			# in case this reduction was cached, load from cache
			Xt = pickle.load(open(cache, 'rb'))
			print('using cached dimensionality reduction data')
		except:
			# do the reduction with given arguments
			print('starting dimensionality reduction')
			beg = process_time()
			Xt = self.reduction(n_components=dims, **self.kwargs).fit_transform(X)
			end = process_time()
			print(f'done in {end-beg:.06f} cputime seconds')

			# cache the result
			pickle.dump(Xt, open(cache, 'wb'))

		# split the reduced points by digit, so that plot will give them different colors
		Xt_by_digit = []
		for digit in range(10):
			Xt_by_digit.append(Xt[y == digit])

		if video:
			print('plotting 3D video animation')
			plot_3d(Xt_by_digit, fname)
		else:
			print('plotting 2D image')
			plot_2d(Xt_by_digit, fname)

class MNIST_PCA(MNIST_DimRed):
	reduction = PCA

class MNIST_SpectralEmbedding(MNIST_DimRed):
	reduction = SpectralEmbedding
	kwargs = {
		'n_jobs': cpu_count(),
		'n_neighbors': 50,
	}

class MNIST_TSNE(MNIST_DimRed):
	reduction = TSNE
	kwargs = {
		'n_jobs': cpu_count(),
		'init': 'random',
		'learning_rate': 'auto',
		'perplexity': 30.0,
	}

class MNIST_UMAP(MNIST_DimRed):
	reduction = UMAP
	kwargs = {
		'n_jobs': cpu_count(),
		'n_neighbors': 15,
	}

methods = [ MNIST_PCA, MNIST_SpectralEmbedding, MNIST_TSNE ]
if UMAP is not None:
	methods.append(MNIST_UMAP)

methods = { m.reduction.__name__: m for m in methods }

parser = argparse.ArgumentParser(description='Utility for plotting MNIST hand-written digits dataset via various dimensionality reduction methods')
parser.add_argument('-n', '--number-of-samples', type=int, default=60000, metavar='N',
		    help='use only the first N samples from the dataset (default: all 60000)')
parser.add_argument('-m', '--method', choices=methods.keys(), required=True,
		    help='the dimensionality reduction method')
parser.add_argument('--neighbors', type=int, metavar='N',
		    help='for methods that support it, use N neighbors instead of the default number')
parser.add_argument('--perplexity', type=float, metavar='P',
		    help='for TSNE method use perplexity P instead of default (30.0)')
parser.add_argument('-v', '--video', action='store_true',
		    help='reduce the dimensionality to 3D instead of 2D and produce a video')
parser.add_argument('-o', '--output', type=str,
		    help='the output file for the image or video')
parser.add_argument('-t', '--time-performance', action='store_true',
		    help='measure time performance of the selected method')

def main():
	args = parser.parse_args()

	if args.output is None and not args.time_performance:
		parser.error('either -o/--output or -t/--time-performance is required')
	elif args.output is not None and args.time_performance:
		parser.error('arguments -o/--output and -t/--time-performance are mutualy exclusive')

	# load the dataset
	(X, y), _ = mnist.load_data()
	del _

	if not args.time_performance:
		# use only requested number of samples
		X = X[:args.number_of_samples, :]
		y = y[:args.number_of_samples]
		print(f'using {args.number_of_samples} from MNIST dataset')

	# reshape images from 28x28 matrices to 784 dimensional vectors
	X = X.reshape(X.shape[0], 28 * 28)

	# prepare arguments for the dimensionality reduction method
	kwargs = { arg: args.__dict__[arg] for arg in ['neighbors', 'perplexity'] if args.__dict__[arg] is not None }

	# load the dimensionality reduction method
	try:
		print(f'using the {args.method} dimensionality reduction method')
		dimred = methods[args.method](**kwargs)
	except Exception as e:
		print(e.args[0])
		exit(1)

	if args.time_performance:
		# measure time performance of the method
		dimred.measure_time_performance(X)
	else:
		# plot image or video
		dimred.plot(X, y, args.output, video=args.video)
		print(f'output saved to file {args.output}')

if __name__ == '__main__':
	main()
