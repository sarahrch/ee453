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
    int return_val;
};

void* SendVals(void* args) { 
    thread_args *recv_args = (thread_args*)args;
    int i = recv_args->index;
    
    // Write data to pipe
    write(fd[i%(SIZE/2)][1], &recv_args, sizeof(thread_args*));
    close(fd[i%(SIZE/2)][1]);

    pthread_exit(NULL);
}

void* MultVals(void* args) {
    thread_args *recv_args = (thread_args*)args;
    int i = recv_args->index;
    int b_val = recv_args->array_val;

    // Read data from pipe -- not really sure what the purpose of piping the data is.. inefficient?
    thread_args *a_args;
    int received = read(fd[i%(SIZE/2)][0], &a_args, sizeof(thread_args*));
    if (received == -1) {
        std::cout << "Error reading from file descriptor" << std::endl;
    }
    int a_val = a_args->array_val;
    close(fd[i%(SIZE/2)][0]);

    delete a_args;
    std::cout << "A args successfully transferred" << std::endl;
    recv_args->return_val = a_val * b_val;
    pthread_exit(&recv_args);
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
        int rcA;
        int rcB;
		rcA = pthread_create(&threadA, NULL, SendVals, args_A);

        if (rcA) {
            std::cout << "Error:unable to create thread," << threadA << std::endl;
            exit(-1);
        } else {
            threads.push_back(std::move(threadA));
        }

        rcB = pthread_create(&threadB, NULL, MultVals, args_B);

        if (rcB) {
            std::cout << "Error:unable to create thread," << threadB << std::endl;
            exit(-1);
        } else {
            threads.push_back(std::move(threadB));
        }
		
	}
    int final;
    for (int i = 0; i < SIZE; i++) {
        void *retval;
        int status = pthread_join(threads[i], &retval);
        if (status) {
            std::cout << "pthread_join failed" << threads[i] << endl;
        }
        if (retval != NULL) {
            thread_args *thread_return = (thread_args*)retval;
            int returned_value = thread_return->return_val;
            final += returned_value;
            delete thread_return;
            std::cout << "Successful accumulation" << std::endl;
        }
    }
    std::cout << "Final multiplication result: " << final << endl;

    return 0;
}