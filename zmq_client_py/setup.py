from setuptools import setup
import os

setup(
	name='clipper_zmq_client',
	version='1.0.0',
	description='A ZMQ-based client for querying Clipper',
	maintainer='Corey Zumar',
	maintainer_email='czumar@berkeley.edu',
	url='http://clipper.ai',
	packages=[
		"clipper_zmq_client"
	],
	install_requires=[
		'numpy',
		'zmq',
		'futures_then'
	])