#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

#include <k_means.cu>

#define h  800 
#define w  800

#define input_file  "input.raw"
#define output_file "output.raw"

int main(int argc, char** argv){
    int i;
    FILE *fp;

  	unsigned char *a = (unsigned char*) malloc (sizeof(unsigned char)*h*w);
    
	// the matrix is stored in a linear array in row major fashion
	if (!(fp=fopen(input_file, "rb"))) {
		printf("can not opern file\n");
		return 1;
	}
	fread(a, sizeof(unsigned char), w*h, fp);
	fclose(fp);
    
	// MY CODE STARTS HERE -------
	// measure the start time here
	int size = h * w;

	// Define means
	char cluster1_mean = 0;
	char cluster2_mean = 85;
	char cluster3_mean = 170;
	char cluster4_mean = 255;

	// Need 800x800 = 640000 threads, CUDA GPUs run kernels using blocks of threads that are a multiple of 32 and no greater than 1024 (Compute Compatability 2.x and later) and 512 for 1.x
	// Choosing 1250 blocks and 512 threads per block - Source: https://stackoverflow.com/questions/9985912/how-do-i-choose-grid-and-block-dimensions-for-cuda-kernels
	k_means<<<1250,512>>>(a, cluster1_mean, cluster2_mean, cluster3_mean, cluster4_mean, size);
	
		
	
	// measure the end time here
	
	// print out the execution time here
	

	// MY CODE ENDS HERE --------
	
	if (!(fp=fopen(output_file,"wb"))) {
		printf("can not opern file\n");
		return 1;
	}	
	fwrite(a, sizeof(unsigned char),w*h, fp);
    fclose(fp);
    
    return 0;
}