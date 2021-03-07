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


// Function to dot a row of A with a column of B
__global__
void dot(int n, float *a, float *b) {

    for (int i = 0; i < n; i ++) {
        c[blockIdx.x][threadIdx.x] += a[blockIdx.x][i] * b[i][threadIdx.x];
    }
}

int main(void) {
    #define N = 1024;

    clock_t start,end;
	start = clock();

    /*
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
    } */
    // Initialize a, b, c

    // TODO: CHANGE THIS TO JUST BE CONTIGUOUS ARRAY??

    float** A = new float*[N];
    // Allocate entire 1024x1024 array in contiguous mem
    A[0] = new float[N * N];
    // Assign a pointer to each new "row" in contiguous mem
    for (int i = 1; i < N; ++i) {
        A[i] = A[i-1] + N;
    }

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A[i][j] = 1.0f;
            B[i][j] = 2.0f;
        }
    }

    // get pointer to allocated device (GPU) memory
    float *dA;
    cudaMalloc((void **)&dA, sizeof(float) * N * N);
    float *dB;
    cudaMalloc((void **)&dB, sizeof(float) * N * N);

    // copy host memory to device (pointing at A[0])
    cudaMemcpy(dA, A[0], sizeof(float) * N * N, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B[0], sizeof(float) * N * N, cudaMemcpyHostToDevice);

    // Set dimensions
    dim3 grid(32,32); // Grid = 32x32 blocks
    dim3 block(32,32); // Block = 32x32 threads


    // Run kernel -- 32x32 threads per block and 32x32 blocks per grid
    dot<<<grid,block>>>(N, dA, dB);

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