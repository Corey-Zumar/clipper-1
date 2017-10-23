from setuptools import setup
import os

setup(
	name='tf_serving_utils',
	version='1',
	description='utilities for benchmarking models in Tensorflow Serving',
	maintainer='Corey Zumar',
	maintainer_email='czumar@berkeley.edu',
	url='http://clipper.ai',
	packages=[
		"tf_serving_utils"
	],
	install_requires=[
		'numpy'
	])