# -*- coding: utf-8 -*-

"""Top-level package for syndi-benchmark."""

__author__ = 'MIT Data To AI Lab'
__email__ = 'dailabmit@gmail.com'
__version__ = '0.1.0.dev0'

import os

import syndi_benchmark.task
import syndi_benchmark.sampler
import syndi_benchmark.task_evaluator
import syndi_benchmark.benchmark

_BASE_PATH = os.path.abspath(os.path.dirname(__file__))

__all__ = (
    'task',
    'sampler',
    'task_evaluator',
    'benchmark'
)