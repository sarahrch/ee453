/*
In this assignment, your task is to multiply the elements of two 1024x1024 arrays (e.g. C[1024x1024]=a[1024x1024]x B[1024x1024]):

-Use 1024*1024 threads
-32x32 threads/block (2-D)
-32x32 blocks/grid (2-D)
-The value of each element of A is 1
-The value of each element of B is 2
-Name this program as 'p2.cu'
-Report the execution time in the report
-After computation, print the value of C[453][453] 
*/

#include <iostream>
#include <math.h>
#include <time.h> //--https://www.tutorialspoint.com/c_standard_library/time_h.htm
#include <cuda_runtime.h> //--https://developer.nvidia.com/blog/even-easier-introduction-cuda and https://docs.nvidia.com/cuda/index.html

// 8 blocks, 512 threads
#define GRIDSIZE = 1
#define NUMBLOCKS = 32
#define BLOCKSIZE = 32

// Function to compute the dot product of a row of A with a col of B
__global__
void dot(int n, float *a, float *b, float *c) {
    c = 0.0f;
    for (int i = 0; i < n; i++) {
        c += a[i] * b[i];
    }
}