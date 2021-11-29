.. _benchmark:

============
Benchmark
============

We provide a benchmarking framework to enable users to compare multiple synthetic data generators against each other. The evaluation metrics are documented within task_evaluator, please visit :ref:`task_evaluator` to read more about it.


Process
-------

We evaluate the performance of pipelines by following a series of executions. From a high level, we can view the process as:

1. Generate a list of tasks of interest.
2. Compute the scores on our test data using multiple metrics (e.g. accuracy and f1).
3. Finally, we output a results csv with these metrics

Benchmark function
~~~~~~~~~~~~~~~~~~
