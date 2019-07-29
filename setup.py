#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='VisualAttribution',
      version='0.1',
      description='Visual Attribution for PyTorch (based on work of Yulong Wang, see https://github.com/yulongwang12/visual-attribution)',
      author='Arne Binder',
      author_email='arne.b.binder@gmail.com',
      url='https://github.com/ArneBinder/visual-attribution',
      packages=find_packages(),
      py_modules=['create_explainer']
     )
