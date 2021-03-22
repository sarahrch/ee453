/*
K-means is a clustering algorithm that is widely used in signal processing, machine learning, etc.
In this problem, you will implement the K-means algorithm to cluster a matrix into K clusters. K-Means algorithm has the following steps:
1. Initialize a mean value for each cluster. 
2. For each data element, compute its 'distance' to the mean value of each cluster. Assign it to the closest cluster.
3. After each data element is assigned to a cluster, recompute the mean value of each cluster. 
4. Check convergence; if the algorithm converges, replace the value of each data with the mean value of the cluster which the data belongs to, then terminate the algorithm; otherwise, go to step 2. 

In this problem, you need to use CUDA programming to parallel compute the "distance" (you need to use 800x800 threads).
The input data is an 800 x 800 matrix which is stored in the 'input.raw'; the value of each matrix element ranges from 0 to 255.
Thus, the matrix can be displayed as an image shown in the following Figure （figure.png）. We will have 4 clusters (K=4).
The initial mean values for the clusters are 0, 85, 170, and 255, respectively.
To simplify the implementation, you do not need to check the convergence;
    run 30 iterations (Step 2 and 3) serially then terminate the algorithm and output the matrix into the le named `output.raw'.
You can refer to the given program `problem1.c' to read the input file and write the output file. It copies the input matrix to the out file directly without any modification.
Submit your program and the report.
In the report, you need to include:
    -the execution time for the 30 iterations (excluding the read/write time)
    -the corresponding image of the output matrix
You can display the output image by using the given Matlab script file.
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <iostream>
#include <cuda_runtime.h> //--https://developer.nvidia.com/blog/even-easier-introduction-cuda and https://docs.nvidia.com/cuda/index.html

//#include <k_means.cu>

#define h  800 
#define w  800

#define input_file  "input.raw"
#define output_file "output.raw"

using namespace std;

__global__
void k_means(unsigned char *a, int *cluster, int mean_c1, int mean_c2, int mean_c3, int mean_c4) {
    // Calculate the distances from each value to current means to find which cluster is closest, stores cluster of index in cluster array
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int min_dist = abs((int)a[index] - mean_c1);
    cluster[index] = 1;
    if (abs((int)a[index] - mean_c2) < min_dist) {
        min_dist = abs((int)a[index] - mean_c2);
        cluster[index] = 2;
    }
    if (abs((int)a[index] - mean_c3) < min_dist) {
        min_dist = abs((int)a[index] - mean_c3);
        cluster[index] = 3;
    }
    if (abs((int)a[index] - mean_c4) < min_dist) {
        min_dist = abs((int)a[index] - mean_c4);
        cluster[index] = 4;
    }
}

int main(int argc, char** argv){
	// GIVEN CODE STARTS HERE ----
    FILE *fp;

  	unsigned char *a; // CHANGING THIS FROM GIVEN CODE TO AVOID MEMORY ACCESS ISSUES = (unsigned char*) malloc (sizeof(unsigned char)*h*w);
	cudaMallocManaged(&a, sizeof(unsigned char)*h*w);

	// the matrix is stored in a linear array in row major fashion
	if (!(fp=fopen(input_file, "rb"))) {
		printf("can not opern file\n");
		return 1;
	}
	fread(a, sizeof(unsigned char), w*h, fp);
	fclose(fp);
	// GIVEN CODE ENDS HERE ----

	// measure the start time here
	clock_t start,end;
	start = clock();

	int size = h * w;

	// Define initial means
	int cluster1_mean = 0;
	int cluster2_mean = 85;
	int cluster3_mean = 170;
	int cluster4_mean = 255;

	int *clusters;
	cudaMallocManaged(&clusters, size*sizeof(int));

	// Initialize all clusters to 0
	for (int i = 0; i < size; i++) {
		clusters[i] = 0;
	}

	// Run 30 iterations of k-means
	for (int i = 0; i < 30; i++) {
		// Need 800x800 = 640000 threads, CUDA GPUs run kernels using blocks of threads that are a multiple of 32 and no greater than 1024 (Compute Compatability 2.x and later) and 512 for 1.x
		// Choosing 1250 blocks and 512 threads per block - Source: https://stackoverflow.com/questions/9985912/how-do-i-choose-grid-and-block-dimensions-for-cuda-kernels
		k_means<<<1250,512>>>(a, clusters, cluster1_mean, cluster2_mean, cluster3_mean, cluster4_mean);

		cudaDeviceSynchronize();

		// Initialize values for calculating new means
		int c1_total = 0;
		int c2_total = 0;
		int c3_total = 0;
		int c4_total = 0;
		int c1_vals = 0;
		int c2_vals = 0;
		int c3_vals = 0;
		int c4_vals = 0;

		for (int j = 0; j < size; j++) {
			if (clusters[j] == 1) {
				c1_total += (int)a[j];
				c1_vals++;
			} else if (clusters[j] == 2) {
				c2_total += (int)a[j];
				c2_vals++;
			} else if (clusters[j] == 3) {
				c3_total += (int)a[j];
				c3_vals++;
			} else if (clusters[j] == 4) {
				c4_total += (int)a[j];
				c4_vals++;
			} else {
				std::cout << clusters[j] << std::endl;
			}
		}
		cluster1_mean = c1_total/c1_vals;
		cluster2_mean = c2_total/c2_vals;
		cluster3_mean = c3_total/c3_vals;
		cluster4_mean = c4_total/c4_vals;
		// FOR DEBUGGING:
		// std::cout << "Means for iteration " << i << ": c1=" << cluster1_mean << " c2= " << cluster2_mean << " c3= " << cluster3_mean << " c4= " << cluster4_mean << std::endl;
	}	

	// Write to output.raw

	// GIVEN CODE STARTS HERE ----
	if (!(fp=fopen(output_file,"wb"))) {
		printf("can not opern file\n");
		return 1;
	}	
	fwrite(a, sizeof(unsigned char),w*h, fp);
    fclose(fp);
	// GIVEN CODE ENDS HERE ----

	// Free Mem
    cudaFree(clusters);
	cudaFree(a);
	
	// measure the end time here
    end = clock();
    double timeElapsed = ((double)((end-start)))/(double)(CLOCKS_PER_SEC);

	// print out the execution time here
    std::cout << "Time: " << timeElapsed << " seconds" << std::endl;
    
    return 0;
}