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

// 32x32 threads per block and 32x32 blocks per grid
#define GRIDSIZE = 1
#define NUMBLOCKS = 1024
#define BLOCKSIZE = 1024

// Function to dot a row of A with a column of B
__global__
void dot(int n, float **a, float **b, float **c) {

    for (int i = 0; i < n; i ++) {
        c[blockIdx.x][threadIdx.x] += a[blockIdx.x][i] * b[i][threadIdx.x];
    }
}

int main(void) {
    int N = 1024;
    float **a, **b, **c;

    clock_t start,end;
	start = clock();

    // Unified Mem Allocation
    cudaMallocManaged(&&a, N*sizeof(float*));
    for (int i = 0; i < N; i++) {
        cudaMallocManaged(a[i], N*sizeof(float)); 
    }  
    cudaMallocManaged(&&b, N*sizeof(float*));
    for (int i = 0; i < N; i++) {
        cudaMallocManaged(b[i], N*sizeof(float)); 
    } 
    cudaMallocManaged(&&c, N*sizeof(float*));
    for (int i = 0; i < N; i++) {
        cudaMallocManaged(c[i], N*sizeof(float)); 
    } 

    // Initialize a, b, c
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            a[i][j] = 1.0f;
            b[i][j] = 2.0f;
            c[i][j] = 0.0f;
        }
    }

    // Run kernel
    dot<<< NUMBLOCKS, BLOCKSIZE >>>(N, a, b);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Error Checking
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            maxError = fmax(maxError, fabs(c[i][j]-2.0f));
        }
    }
    std::cout << "Max Error: " << maxError << std::endl;

    // Print C[453][453]
    std::cout << c[453][453] << std::endl;

    // Free Mem
    for(int i = 0; i < N; i++) {
        cudaFree(a[i]);
    } 
    cudaFree(a); 
    for(int i = 0; i < N; i++) {
        cudaFree(b[i]);
    }  
    cudaFree(b);
    for(int i = 0; i < N; i++) {
        cudaFree(c[i]);
    }  
    cudaFree(c);

    end = clock();
    double timeElapsed = ((double)((end-start)))/(double)(CLOCKS_PER_SEC);
    std::cout << "Time: " << timeElapsed << " seconds" << std::endl;

    return 0;
}