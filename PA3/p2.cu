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
void dot(int N, float *a, float *b, float *c) {
    // Find thread position
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate dot of a row with a col
    float ans = 0;
    for (int i = 0; i < N; i++) {
        ans += a[N * row + i] * b[i * N + col];
    }

    // Store in matrix c
    c[row * N + col] = ans;
}

int main() {
    int N = 1024;

    clock_t start,end;
	start = clock();

    float *a, *b, *c;

    // Allcoate 1024x1024 matrix as 1-D array in unified memory
    cudaMallocManaged(&a, N*N*sizeof(float));
    cudaMallocManaged(&b, N*N*sizeof(float));
    cudaMallocManaged(&c, N*N*sizeof(float));

    // Initialize matrices
    for (int i = 0; i < N*N; i++)
    {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    // Set dimensions
    dim3 grid(32,32); // Grid = 32x32 blocks
    dim3 block(32,32); // Block = 32x32 threads

    // Run kernel -- 32x32 threads per block and 32x32 blocks per grid
    dot<<<grid,block>>>(N, a, b, c);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Error Checking
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            maxError = fmax(maxError, fabs(c[i * N + j] - 2048.0f));
        }
    }
    std::cout << "Max Error: " << maxError << std::endl;

    // Print C[453][453]
    std::cout << c[453 * N + 453] << std::endl;

    // Free Mem
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    end = clock();
    double timeElapsed = ((double)((end-start)))/((double)(CLOCKS_PER_SEC));
    std::cout << "Time: " << timeElapsed << " seconds" << std::endl;

    return 0;
}