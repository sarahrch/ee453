#include <iostream>
#include <vector>
#include <pthread.h>
#include <unistd.h>
using namespace std;

/*
Assignment: These threads share the process's resources but are able to execute independently.
In this problem, you will implement a 10 threads C/C++ program to parallel compute matrix multiplication between a 1x5 matrix (matrix A) and 5x1 matrix (matrix B).
You need to do the following steps (A and B are two matrixes of all fives)

1. Create 10 threads and assign the matrixes elements into each thread (one for each).
For example, assign the first element of A in thread 1, the second in thread 2, ...; assign the first element of B in thread 6, the second in thread 7...
2. After each data element is assigned to each thread, use pipes to send the elements from one to another and calculate the multiplication results between these two.
For example, send the element in thread 1 (the first element in A) to thread 6 and calculate the multiplication result. 
3. Add these 5 multiplication results together. 
4. Record execution time and calculation result in the report.
*/

// Must be run on Linux system

void SendVals(int val, int &fd) {
    // Not reading
    close(fd[0]);
    // Write data to pipe
    write(fd[1], val);
    close(fd[1]);
    pthread_exit();
}

int MultVals(int val, int &fd) {
    // Not writing
    close(fd[1]);
    // Read data from pipe
    int received = read(fd[0]);
    close(fd[0]);
    pthread_exit(val * received);
}

int main(int argc, char** argv) {

    int NUM_THREADS = 10;
    // Take A, B value input ? <- see if we need to do this

    // Create threads and send A, B values
    std::vector <pthread_t> threads;
	
	for(int i = 0; i < NUM_THREADS/2; i++) {
        // Create pipe
        int fd[2];
        status = pipe(fd);
        if (status == -1) {
           std::cout << "Error:unable to create pipe" << endl;
        }

        // Create threads
        pthread_t threadA, threadB;
		t1 = pthread_create(&threadA, NULL, SendVals, A[I], &fd);

        if (t1) {
            cout << "Error:unable to create thread," << t1 << endl;
            exit(-1);
        } else {
            threads.push_back(std::move(t1));
        }

        t2 = pthread_create(&threadB, NULL, MultVals, B[I], &fd);

        if (t2) {
            cout << "Error:unable to create thread," << t2 << endl;
            exit(-1);
        } else {
            threads.push_back(std::move(t2));
        }
		
	}
    int final;
    for (int i = 0; i < NUM_THREADS; i++) {
        void *res;
        int status = pthread_join(threads[i], &res);
        if (status == -1) {
            std::cout << "pthread_join failed" << threads[i] << endl;
        }
        final = final + *res;
    }
    std::cout << "Final multiplication result: " << final << endl;

    return 0;
}