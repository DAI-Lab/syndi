.. _sampler:

====
Sampler
====

Sampler takes a task as input and the corresponding data generator. Sampler uses the data generator to make synthetic data that it concatenates with the training data and returns.

Data Format
-----------

Input
~~~~~

Here we will have an example table, train_data, generator.

* ``timestamp``: an INTEGER or FLOAT column with the time of the observation in Unix Time Format
* ``value``: an INTEGER or FLOAT column with the observed value at the indicated timestamp

This is an example of such table:

+------------+-----------+
|  timestamp |     value |
+------------+-----------+
| 1222819200 | -0.366358 |
+------------+-----------+
| 1222840800 | -0.394107 |
+------------+-----------+
| 1222862400 |  0.403624 |
+------------+-----------+
| 1222884000 | -0.362759 |
+------------+-----------+
| 1222905600 | -0.370746 |
+------------+-----------+

Synthetic Data Generation
~~~~~~~~~~~~~~~~~~~~~~~~~

Here we will show that the sampler uses the task defined parameters to generate the synthetic data.

Output
~~~~~~

Here we will show that the output is simply synthetic data concatenated with the training data.:

* ``start``: timestamp where the anomalous interval starts
* ``end``: timestamp where the anomalous interval ends

Optionally, a third column called ``severity`` can be included with a value that represents the
severity of the detected anomaly.

An example of such a table is:

+------------+------------+----------+
|      start |        end | severity |
+------------+------------+----------+
| 1222970400 | 1222992000 | 0.572643 |
+------------+------------+----------+
| 1223013600 | 1223035200 | 0.572643 |
+------------+------------+----------+

