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

__global__
void k_means(unsigned char *a, char mean_c1, char mean_c2, char mean_c3, char mean_c4, int size) {

}