#include <iostream>
#include <vector>
#include <pthread.h>
using namespace std;

void AddIVals(int &val1, int val2) {

}

int main(int argc, char** argv) {

    int NUM_THREADS = 10;
    // Take A, B value input ? <- see if we need to do this

    // Create threads and send A, B values
    std::vector <pthread_t> threads;
	
	for(int i = 0; i < NUM_THREADS/2; i++) {
        pthread_t thread;
		t = pthread_create(&thread, NULL, AddIVals, A[I]);

        if (t) {
            cout << "Error:unable to create thread," << t << endl;
            exit(-1);
        } else {
            threads.push_back(std::move(t));
        }
		
	}
    for (int i = NUM_THREADS/2; i < NUM_THREADS; i++) {
        t = pthread_create(&thread, NULL, AddIVals, B[I]);

        if (t) {
            cout << "Error:unable to create thread," << t << endl;
            exit(-1);
        } else {
            threads.push_back(std::move(t));
        }
        

    }


}