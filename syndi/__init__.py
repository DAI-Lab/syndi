# -*- coding: utf-8 -*-

"""Top-level package for syndi."""

__author__ = 'MIT Data To AI Lab'
__email__ = 'dailabmit@gmail.com'
__version__ = '0.1.0.dev0'

import os

import syndi.task
import syndi.sampler
import syndi.task_evaluator
import syndi.benchmark

_BASE_PATH = os.path.abspath(os.path.dirname(__file__))

__all__ = (
    'task',
    'sampler',
    'task_evaluator',
    'benchmark'
)