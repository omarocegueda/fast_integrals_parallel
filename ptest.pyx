cimport cython
import numpy as np
cimport numpy as cnp
from cython.parallel import parallel, threadid, prange
cimport openmp
from libc.stdio cimport printf
from libc.stdlib cimport abort, malloc, free
from dipy.utils.omp cimport set_num_threads, restore_default_num_threads


def test_func():
    cdef int thread_id = -1
    with nogil, parallel(num_threads=10):
        thread_id = threadid()
        printf("Thread ID: %d\n", thread_id)
        
        
cdef double*** allocate_volume(int n0, int n1, int n2)nogil:
    cdef:
        double ***mem
        cnp.npy_intp i, j, k
        
    mem = <double***>malloc(sizeof(double**) * n0)
    for i in range(n0):
        mem[i] = <double**>malloc(sizeof(double*) * n1)
        for j in range(n1):
            mem[i][j] = <double*>malloc(sizeof(double) * n2)
            for k in range(n2):
                mem[i][j][k] = 0    
    return mem               
                    
                    
cdef void free_volume(double ***mem, int n0, int n1)nogil:
    cdef:
        cnp.npy_intp i, j
    
    for i in range(n0):
        for j in range(n1):
            free(mem[i][j])
        free(mem[i])
    free(mem)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def rectangle_sums_sequential(double[:,:,:] I, int[:] m, double[:,:,:] out):
    cdef:
        cnp.npy_intp n0 = I.shape[0]
        cnp.npy_intp n1 = I.shape[1]
        cnp.npy_intp n2 = I.shape[2]
        cnp.npy_intp m0 = m[0]
        cnp.npy_intp m1 = m[1]
        cnp.npy_intp m2 = m[2]
        cnp.npy_intp i, j, k, prev_i, cur_i
        double integral
        double ***T
    with nogil:
        T = allocate_volume(2, n1, n2)
        cur_i = 1
        for i in range(n0):
            cur_i = 1 - cur_i
            prev_i = 1 - cur_i
            for j in range(n1):
                for k in range(n2):
                    # Start with last corner
                    integral = I[i,j,k] # q=(0,0,0)
                    # Add signed rectangles
                    if i>0:
                        integral += T[prev_i][j][k]# q=(1, 0, 0)
                        if j>0:
                            integral -= T[prev_i][j-1][k]# q=(1, 1, 0)
                            if k>0:
                                integral += T[prev_i][j-1][k-1]# q=(1, 1, 1)
                        if k>0:
                            integral -= T[prev_i][j][k-1]# q=(1, 0, 1)
                    if j>0:
                        integral += T[cur_i][j-1][k]# q=(0, 1, 0)
                        if k>0:
                            integral -= T[cur_i][j-1][k-1]# q=(0, 1, 1)
                    if k>0:
                        integral += T[cur_i][j][k-1]# q=(0, 0, 1)
                        
                    # Add displaced signed corners
                    if i>=m0:
                        integral -= I[i-m0,j,k]# q=(1, 0, 0)
                        if j>=m1:
                            integral += I[i-m0,j-m1,k]# q=(1, 1, 0)
                            if k>=m2:
                                integral -= I[i-m0,j-m1,k-m2]# q=(1, 1, 1)
                        if k>=m2:
                            integral += I[i-m0,j,k-m2]# q=(1, 0, 1)
                    if j>=m1:
                        integral -= I[i,j-m1,k]# q=(0, 1, 0)
                        if k>=m2:
                            integral += I[i,j-m1,k-m2]# q=(0, 1, 1)
                    if k>=m2:
                        integral -= I[i,j,k-m2]# q=(0, 0, 1)
                        
                    # Save current integral for future reference
                    T[cur_i][j][k] = integral
                    
                    # Use integral of current rectangle
                    out[i,j,k] = integral
        free_volume(T, 2, n1)  
                    
                    
                    

    
                        
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def rectangle_sums_parallel(double[:,:,:] I, int[:] m, double[:,:,:] out, int num_threads):
    cdef:
        cnp.npy_intp n0 = I.shape[0]
        cnp.npy_intp n1 = I.shape[1]
        cnp.npy_intp n2 = I.shape[2]
        cnp.npy_intp m0 = m[0]
        cnp.npy_intp m1 = m[1]
        cnp.npy_intp m2 = m[2]
        cnp.npy_intp i, j, k, prev_i, cur_i, s_padded, s_start, s_end, s_per_thread, thread_id
        double*** T
        double *integral
    set_num_threads(num_threads)
    with nogil, parallel():
        num_threads = openmp.omp_get_num_threads()
        thread_id = threadid()
        T = allocate_volume(2, n1, n2)
        integral = <double*>malloc(sizeof(double))
        
        # Assign a consecutive range of slices to this thread
        s_per_thread = (n0 + num_threads - 1) / num_threads
        s_start = thread_id * s_per_thread
        s_end = s_start + s_per_thread
        if s_end > n0:
            s_end = n0
        # Compute overlapping ranges: in order to compute a slice, we need previous m[0] slices
        s_padded = s_start - m[0]
        if s_padded < 0:
            s_padded = 0 

        cur_i = 1
        for i in range(s_padded, s_end):
            cur_i = 1 - cur_i
            prev_i = 1 - cur_i
            for j in range(n1):
                for k in range(n2):
                    # Start with last corner
                    integral[0] = I[i,j,k] # q=(0,0,0)
                    # Add signed rectangles
                    if i>s_padded: # Our "zero" is now the first slice we are processing in this thread
                        integral[0] += T[prev_i][j][k]# q=(1, 0, 0)
                        if j>0:
                            integral[0] -= T[prev_i][j-1][k]# q=(1, 1, 0)
                            if k>0:
                                integral[0] += T[prev_i][j-1][k-1]# q=(1, 1, 1)
                        if k>0:
                            integral[0] -= T[prev_i][j][k-1]# q=(1, 0, 1)
                    if j>0:
                        integral[0] += T[cur_i][j-1][k]# q=(0, 1, 0)
                        if k>0:
                            integral[0] -= T[cur_i][j-1][k-1]# q=(0, 1, 1)
                    if k>0:
                        integral[0] += T[cur_i][j][k-1]# q=(0, 0, 1)
                        
                    # Add displaced signed corners
                    if i>=s_padded + m0: # Our "zero" is now the first slice we are processing in this thread
                        integral[0] -= I[i-m0,j,k]# q=(1, 0, 0)
                        if j>=m1:
                            integral[0] += I[i-m0,j-m1,k]# q=(1, 1, 0)
                            if k>=m2:
                                integral[0] -= I[i-m0,j-m1,k-m2]# q=(1, 1, 1)
                        if k>=m2:
                            integral[0] += I[i-m0,j,k-m2]# q=(1, 0, 1)
                    if j>=m1:
                        integral[0] -= I[i,j-m1,k]# q=(0, 1, 0)
                        if k>=m2:
                            integral[0] += I[i,j-m1,k-m2]# q=(0, 1, 1)
                    if k>=m2:
                        integral[0] -= I[i,j,k-m2]# q=(0, 0, 1)
                        
                    T[cur_i][j][k] = integral[0]
                    # Use integral of current rectangle
                    if (s_start <= i) and (i < s_end):
                        out[i,j,k] = integral[0]
        free_volume(T, 2, n1)                
                        
                       
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)                       
cdef double integrate_rectangle(double[:,:,:] I, int i, int j, int k, int[:] m)nogil:    
    cdef:
        cnp.npy_intp m0 = m[0]
        cnp.npy_intp m1 = m[1]
        cnp.npy_intp m2 = m[2]
        cnp.npy_intp ii, jj, kk
        double s
    s = 0
    for ii in range(m0):
        if ii>i:
            continue
        for jj in range(m1):
            if jj>j:
                continue
            for kk in range(m2):
                if kk<=k:
                    s += I[i-ii,j-jj,k-kk]
    return s
    
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def rectangle_sums_direct(double[:,:,:] I, int[:] m, double[:,:,:] out):
    cdef:
        cnp.npy_intp n0 = I.shape[0]
        cnp.npy_intp n1 = I.shape[1]
        cnp.npy_intp n2 = I.shape[2]
        cnp.npy_intp m0 = m[0]
        cnp.npy_intp m1 = m[1]
        cnp.npy_intp m2 = m[2]
        cnp.npy_intp i, j, k, prev_i, cur_i
        double integral
    with nogil:
        for i in range(n0):
            for j in range(n1):
                for k in range(n2):
                    integral = integrate_rectangle(I, i, j, k, m)
                    # Use integral of current rectangle
                    out[i,j,k] = integral              
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                
