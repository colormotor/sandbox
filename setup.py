from setuptools import setup, find_packages
import sys

setup(name='sandbox',
        version='0.1',
        description='Colormotor sandbox utilities',
        url='',
        author='Daniel Berio',
        author_email='drand48@gmail.com',
        license='MIT',
        packages=find_packages(),
        install_requires = ['numpy','scipy','matplotlib'],
        zip_safe=False)
