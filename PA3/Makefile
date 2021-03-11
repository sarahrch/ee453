CXX=g++
NVCC=nvcc
CPPFLAGS = -g -Wall -std=c++11

all: p1 p2

p1: p1.cu
	$(NVCC) $(CPPFLAGS) $^ -o $@

p2: p2.cu
	$(NVCC) $(CPPFLAGS) $^ -o $@

.PHONY: clean

clean:
	rm -f p1 *.o
	rm -f p2 *.o