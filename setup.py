#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md', encoding='utf-8') as history_file:
    history = history_file.read()

install_requires = [
    's3fs>=0.2.2',
    'numpy>=1.18',
    'pandas>=1.1',
    'scikit-learn>=0.23.1',
    'pycaret>=2.3.0',
    'sdv>=0.9.0',
    'pyts>=0.9',

    # fix conflict
    'h5py>=2.10.0',
    'urllib3>=1.21.1',
    'matplotlib>=3.3.3'
    ]

setup_requires = [
    'pytest-runner>=2.11.1',
]

tests_require = [
    'pytest>=3.4.2',
    'pytest-cov>=2.6.0',
    'rundoc>=0.4.3,<0.5',
]

development_requires = [
    # general
    'bumpversion>=0.5.3',
    'pip>=9.0.1',
    'watchdog>=0.8.3',

    # docs
    'm2r2>=0.2.5,<0.3',
    'nbsphinx>=0.5.0,<0.7',
    'Sphinx>=3,<3.3',
    'pydata-sphinx-theme',
    'autodocsumm>=0.1.10,<1',
    'ipython>=6.5,<7.5',

    # style check
    'flake8>=3.7.7',
    'isort>=4.3.4',

    # fix style issues
    'autoflake>=1.2',
    'autopep8>=1.4.3',

    # distribute on PyPI
    'twine>=1.10.0',
    'wheel>=0.30.0',

    # Advanced testing
    'coverage>=4.5.1',
    'tox>=2.9.1',

    # Documentation style
    'pydocstyle==3.0.0,<4',
]

setup(
    author='MIT Data To AI Lab',
    author_email='dailabmit@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description='e2e pipeline to generate synthetic data for empowering machine learning models',
    entry_points={
        'console_scripts': [
            'syndi=syndi.cli:main',
        ],
    },
    extras_require={
        'test': tests_require,
        'dev': development_requires + tests_require,
    },
    install_package_data=True,
    install_requires=install_requires,
    license='MIT license',
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/markdown',
    include_package_data=True,
    keywords='syndi syndi syndi',
    name='syndi',
    packages=find_packages(include=['syndi', 'syndi.*']),
    python_requires='>=3.6',
    setup_requires=setup_requires,
    test_suite='tests',
    tests_require=tests_require,
    url='https://github.com/DAI-Lab/syndi',
    version='0.1.0.dev0',
    zip_safe=False,
)
