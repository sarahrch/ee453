CXX = g++
CPPFLAGS = -g -Wall -std=c++11

matrixmult: multithread_matrixmult.cpp
	$(CXX) $(CPPFLAGS) $^ -pthread -o matrixmult

.PHONY: clean

clean:
	rm -f matrixmult *.o