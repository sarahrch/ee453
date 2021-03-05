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
#include <time.h> //--https://www.tutorialspoint.com/c_standard_library/time_h.htm
#include <cuda_runtime.h> //--https://developer.nvidia.com/blog/even-easier-introduction-cuda and https://docs.nvidia.com/cuda/index.html
#include <device_launch_parameters.h> //--https://stackoverflow.com/questions/6061565/setting-up-visual-studio-intellisense-for-cuda-kernel-calls


// Function to add each element
__global__
void array_add(int n, float *a, float *b) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        b[i] = a[i] + b[i];
    }
}

int main(void) {
    int N = 65536;
    float *a, *b;

    clock_t start,end;
	start = clock();

    // Unified Mem Allocation
    cudaMallocManaged(&a, N*sizeof(float));
    cudaMallocManaged(&b, N*sizeof(float));

    // Initialize a and b
    for (int i = 0; i < N; i++)
    {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    // Run kernel with 8 blocks, 512 threads
    array_add<<<8, 512>>>(N, a, b);

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

    end = clock();
    double timeElapsed = ((double)((end-start)))/(double)(CLOCKS_PER_SEC);
    std::cout << "Time: " << timeElapsed << " seconds" << std::endl;

    return 0;
}