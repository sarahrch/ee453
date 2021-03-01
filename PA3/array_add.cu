/*
In this assignment, your task is to add the elements of two 1x65536 arrays (e.g. C[65536]=a[65536]+B[65536]):

-Use 4096 threads (each thread calculates 65536/4096=16 elements)
-512 threads/block (1-D)
-8 blocks (1-D)
-The value of each element of A is 1
-The value of each element of B is 2
-Name this program as 'p1.cu'
-Report the execution time in the report
-After computation, print the first 256 values of the C matrix.
*/

#include <iostream>
#include <math.h>
#include <cuda_runtime.h>

// Function to add each element
__global__
void array_add(int n, float *a, float *b) {
    for (int i = 0; i < n; i++) {
        b[i] = a[i] + b[i];
    }
}

int main(void) {
    int N = 65536;
    float *a, *b;

    // Unified Mem Allocation
    cudaMallocManaged(&a, N*sizeof(float));
    cudaMallocManaged(&b, N*sizeof(float));

    // Initialize x and y
    for (int i = 0; i < N; i++)
    {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    // Run kernel: 8 blocks, 512 threads
    array_add<<< 8, 512 >>>(N, a, b);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Error Checking
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
    {
        maxError = fmax(maxError, fabs(b[i]-3.0f));
    }
    std::cout << "Max Error: " << maxError << std::endl;

    // Print first 256 values
    for (int i = 0; i < 256; i++)
    {
        std::cout << b[i] << std::endl;
    }

    // Free Mem
    cudaFree(a);
    cudaFree(b);

    return 0;
}