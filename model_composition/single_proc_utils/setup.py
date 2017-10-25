from setuptools import setup
import os

setup(
	name='single_proc_utils',
	version='1',
	description='utilities for benchmarking models used in Clipper model composition research',
	maintainer='Corey Zumar',
	maintainer_email='czumar@berkeley.edu',
	url='http://clipper.ai',
	packages=[
		"single_proc_utils", "single_proc_utils.model", "single_proc_utils.driver_utils"
	],
	install_requires=[
		'numpy',
		'Pillow'
	])
