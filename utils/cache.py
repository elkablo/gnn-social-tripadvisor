#!/usr/bin/env python3

import pickle
from os import mkdir
from os.path import isdir, dirname, basename, join
import inspect

def cache(*relevant_args):
	'A simple cache module that caches return values of function calls'

	relevant_fargs = list(filter(lambda x: x[0:2] != '--', relevant_args))
	relevant_cmdline_args = list(map(lambda x: x[2:], filter(lambda x: x[0:2] == '--', relevant_args)))
	relevant_cmdline_args.sort()

	def decorator(func):
		def f(*fargs, **fkwargs):
			import __main__

			if not hasattr(__main__, 'args') or not hasattr(__main__.args, 'cache') or not __main__.args.cache:
				return func(*fargs, **fkwargs)

			func_args = inspect.getfullargspec(func).args
			fargs_repr = []
			for i, argname in zip(range(len(func_args)), func_args):
				if argname in relevant_fargs:
					fargs_repr.append(f'{argname}={repr(fargs[i])}')
			fargs_repr = ','.join(fargs_repr)
			cmdline_repr = ','.join([arg + '=' + repr(__main__.args.__dict__[arg]) for arg in relevant_cmdline_args])

			cache_dir = join(dirname(__main__.__file__), '.cache')
			cache_file = f'{basename(__main__.__file__)}.{func.__name__}({fargs_repr})_{cmdline_repr}.bin'
			cache_path = join(cache_dir, cache_file)

			try:
				result = pickle.load(open(cache_path, 'rb'))
			except:
				if not isdir(cache_dir):
					mkdir(cache_dir)
				result = func(*fargs, **fkwargs)
				pickle.dump(result, open(cache_path, 'wb'))

			return result

		return f

	return decorator
