.. _quickstart:

Quickstart
==========

In the following steps we will show a short guide about how to run 
**syndi** on your dataset

1. Generate Tasks
-----------------

We generate a list of Task objects that allow us to record results for different 
**Sampling Methods** and **Regression/Classification** models.

To do so, we need to import the `train_df` and `test_df` dataframes.

.. ipython:: python
    :okwarning:

	print(10*10)

import syndi

   

The output will be a table that contains two columns `timestamp` and `value`.

2. Benchmark Tasks
------------------

Once we have the tasks, we can run them through the benchmarking pipeline to get results.

.. ipython:: python
    :okwarning:
    
    print(3*3)


The output will be a ``pandas.Dataframe`` containing a collection of task metrics.
