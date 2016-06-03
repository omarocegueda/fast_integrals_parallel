import numpy as np
import ptest
from time import time

def test_simple_sums():
    n = np.array([100, 100, 100], dtype=np.int32)
    m = np.array([4, 5, 6], dtype=np.int32)
    I = np.empty(tuple(n), dtype=np.float64)
    I = np.random.random(I.size).reshape(tuple(n))
    
    # First test accuracy of the sequential algorithm
    out_slow = np.zeros(tuple(n), dtype=np.float64)
    out_fast = np.zeros(tuple(n), dtype=np.float64)
    ptest.rectangle_sums_sequential(I, m, out_fast)
    ptest.rectangle_sums_direct(I, m, out_slow)
    print("Maximum difference (direct vs. fast-sequential): %f"%(np.abs(out_fast-out_slow).max(),))
    
    for num_threads in range(1, 5):
        out_fast = np.zeros(tuple(n), dtype=np.float64)
        ptest.rectangle_sums_parallel(I, m, out_fast, num_threads)
        print("Maximum difference (direct vs. fast-parallel[%d threads]): %f"%(num_threads, np.abs(out_fast-out_slow).max(),))

    # Now test performance
    n = np.array([300, 300, 300], dtype=np.int32)
    m = np.array([4, 5, 6], dtype=np.int32)
    I = np.empty(tuple(n), dtype=np.float64)
    I = np.random.random(I.size).reshape(tuple(n))
    out_fast = np.zeros(tuple(n), dtype=np.float64)
    start = time()
    ptest.rectangle_sums_sequential(I, m, out_fast)
    end = time()
    print "Fast sequential. Elapsed: %f"%(end-start,)
    for nthreads in range(1, 5):
        start = time()
        ptest.rectangle_sums_parallel(I, m, out_fast, nthreads)
        end = time()
        print "%d threads. Elapsed: %f"%(nthreads, end-start)

test_simple_sums()
