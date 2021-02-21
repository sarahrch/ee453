#include <iostream>
#include <vector>
#include <pthread.h>
#include <unistd.h>
using namespace std;

// Must be run on Linux system

void SendVals(int val, int &fd) {
    // Not reading
    close(fd[0]);
    // Write data to pipe
    write(fd[1], val);
    close(fd[1]);
    exit(EXIT_SUCCESS);
}

void MultVals(int val, int &fd) {
    // Not writing
    close(fd[1]);
    // Read data from pipe
    int received = read(fd[0]);
    close(fd[0]);
    exit(EXIT_SUCCESS);
}

int main(int argc, char** argv) {

    int NUM_THREADS = 10;
    // Take A, B value input ? <- see if we need to do this

    // Create threads and send A, B values
    std::vector <pthread_t> threads;
	
	for(int i = 0; i < NUM_THREADS/2; i++) {
        pthread_t threadA, threadB;
        int fd[2];
		t1 = pthread_create(&threadA, NULL, SendVals, A[I], &fd);

        if (t1) {
            cout << "Error:unable to create thread," << t << endl;
            exit(-1);
        } else {
            threads.push_back(std::move(t));
        }

        t2 = pthread_create(&threadB, NULL, MultVals, B[I], &fd);

        if (t2) {
            cout << "Error:unable to create thread," << t << endl;
            exit(-1);
        } else {
            threads.push_back(std::move(t));
        }
		
	}


}