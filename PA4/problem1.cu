#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <iostream>

//#include <k_means.cu>

#define h  800 
#define w  800

#define input_file  "input.raw"
#define output_file "output.raw"

__global__
void k_means(unsigned char *a, unsigned char *cluster, unsigned char mean_c1, unsigned char mean_c2, unsigned char mean_c3, unsigned char mean_c4) {
    // Calculate the distances from each value to current means to find which cluster is closest, stores cluster of index in cluster array
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned char min_dist = a[index] - mean_c1;
    cluster[index] = 1;
    if (a[index] - mean_c2 < min_dist) {
        min_dist = a[index] - mean_c2;
        cluster[index] = 2;
    }
    if (a[index] - mean_c3 < min_dist) {
        min_dist = a[index] - mean_c3;
        cluster[index] = 3;
    }
    if (a[index] - mean_c4 < min_dist) {
        min_dist = a[index] - mean_c4;
        cluster[index] = 4;
    }
}

int main(int argc, char** argv){
	// GIVEN CODE STARTS HERE ----
    FILE *fp;

  	unsigned char *a = (unsigned char*) malloc (sizeof(unsigned char)*h*w);
    
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
	unsigned char cluster1_mean = 0;
	unsigned char cluster2_mean = 85;
	unsigned char cluster3_mean = 170;
	unsigned char cluster4_mean = 255;

	unsigned char *clusters;
	cudaMallocManaged(&clusters, size*sizeof(unsigned char));

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
		unsigned char c1_vals = 0;
		unsigned char c2_vals = 0;
		unsigned char c3_vals = 0;
		unsigned char c4_vals = 0;

		for (int j = 0; j < size; j++) {
			if (clusters[i] == 1) {
				c1_total += a[i];
				c1_vals++;
			} else if (clusters[i] == 2) {
				c2_total += a[i];
				c2_vals++;
			} else if (clusters[i] == 3) {
				c3_total += a[i];
				c3_vals++;
			} else if (clusters[i] == 4) {
				c4_total += a[i];
				c4_vals++;
			}
		}

		cluster1_mean = c1_total/c1_vals;
		cluster2_mean = c2_total/c2_vals;
		cluster3_mean = c3_total/c3_vals;
		cluster4_mean = c4_total/c4_vals;
		std::cout << "Means for iteration " << i << ": c1=" << cluster1_mean << " c2= " << cluster2_mean << " c3= " << cluster3_mean << " c4= " << cluster4_mean << std::endl;
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
	
	// measure the end time here
    end = clock();
    double timeElapsed = ((double)((end-start)))/(double)(CLOCKS_PER_SEC);

	// print out the execution time here
    std::cout << "Time: " << timeElapsed << " seconds" << std::endl;
    
    return 0;
}