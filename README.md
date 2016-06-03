To build:

$ python setup.py build_ext -i

To test:

$ python test.py

Sample output using a 2-core computer:
<pre>
First verify that both approaches are equivalent:
Maximum difference (direct vs. fast-sequential): 0.000000
Maximum difference (direct vs. fast-parallel[1 threads]): 0.000000
Maximum difference (direct vs. fast-parallel[2 threads]): 0.000000
Maximum difference (direct vs. fast-parallel[3 threads]): 0.000000
Maximum difference (direct vs. fast-parallel[4 threads]): 0.000000

Now check performance (on a computer with 2 physical cores i5-4200H @ 2.80GHz):
Fast sequential. Elapsed: 0.352472
1 threads. Elapsed: 0.364174<<<< A bit slower than the sequential algorithm
2 threads. Elapsed: 0.196001<<<< We only have 2 physical cores
3 threads. Elapsed: 0.250694
4 threads. Elapsed: 0.200792
</pre>
