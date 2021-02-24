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

// Global
const int SIZE = 10; // Total number of threads (Inner matrix size * 2)
int fd[SIZE/2][2]; // Pipes for each pair of values

struct thread_args {
    int index;
    int array_val;
};

void SendVals(const void* args) { 
    thread_args *recv_args = (thread_args*)args;
    int i = recv_args->index;
    int v = recv_args->array_val;
    
    // Not reading
    close(fd[i%(SIZE/2)][0]);
    // Write data to pipe
    write(fd[i%(SIZE/2)][1], v, sizeof(int));
    close(fd[i%(SIZE/2)][1]);

    delete *args;
    pthread_exit(NULL);
}

void MultVals(const void* args) {
    thread_args *recv_args = (thread_args*)args;
    int i = recv_args->index;
    int v = recv_args->array_val;

    // Not writing
    close(fd[i%(SIZE/2)][1]);
    // Read data from pipe
    void* buf;
    int received = read(fd[i%(SIZE/2)][0], buf, sizeof(int));
    if (received == -1) {
        std::cout << "Error reading from file descriptor" << std::endl;
    }
    int a_val = (*(int*)buf);
    close(fd[i%(SIZE/2)][0]);

    delete *args;
    pthread_exit((void*)(a_val * b_val));
}

int main(int argc, char** argv) {

    // Take A, B value input ? <- see if we need to do this

    int A[SIZE/2] = {1, 2, 3, 4, 5};
    int B[SIZE/2] = {6, 7, 8, 9, 10};

    // Create threads and send A, B values
    std::vector <pthread_t> threads;
	
	for(int i = 0; i < SIZE/2; i++) {
        
        // Create pipe
        int status = pipe(fd[i]);
        if (status == -1) {
           std::cout << "Error:unable to create pipe" << std::endl;
        }

        // Allocate mem for passed values
        thread_args *args_A = new thread_args;
        args_A->index = i;
        args_A->array_val = A[i];

        thread_args *args_B = new thread_args;
        args_B->index = i;
        args_B->array_val = B[i];

        // Create threads
        pthread_t threadA, threadB;
		threadA = pthread_create(&threadA, NULL, SendVals, args_A);

        if (threadA) {
            std::cout << "Error:unable to create thread," << threadA << std::endl;
            exit(-1);
        } else {
            threads.push_back(std::move(threadA));
        }

        threadB = pthread_create(&threadB, NULL, MultVals, args_B);

        if (threadB) {
            std::cout << "Error:unable to create thread," << threadB << std::endl;
            exit(-1);
        } else {
            threads.push_back(std::move(threadB));
        }
		
	}
    long final;
    for (int i = 0; i < SIZE; i++) {
        void *return_val;
        int status = pthread_join(threads[i], &return_val);
        if (status) {
            std::cout << "pthread_join failed" << threads[i] << endl;
        }
        if (return_val != NULL) {
            final = final + (long)return_val; // Tricking compiler? Not sure if this is the way it is conventionally done
        }
    }
    std::cout << "Final multiplication result: " << final << endl;

    return 0;
}