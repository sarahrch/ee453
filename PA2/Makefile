CXX = g++
CPPFLAGS = -g -Wall -std=c++11

multithread: multithread_matrixmult.cpp
	$(CXX) $(CPPFLAGS) $^ -pthread -o multithread

.PHONY: clean

clean:
	rm -f multithread *.o