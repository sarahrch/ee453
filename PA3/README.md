Sarah Chow - 4091216117
EE453 PA 3

# P1 Description:
This program contains two hard-coded 1x65536 arrays which are added together. Both the size of the arrrays and their contents may be modified by editing the code.
The expected values are error checked against their actual values.

Details:
-4096 threads (each thread calculates 65536/4096=16 elements)
-512 threads/block (1-D)
-8 blocks (1-D)
-The value of each element of array A is 1
-The value of each element of array B is 2

Execution Time: 0.168317 seconds

# P2 Description:
This program contains two hard-coded 1024x1024 which are multiplied together. They are each implemented as flattened, 1D arrays.
Both the size of the matrices and their contents may be modified by editing the code. The expected values are error checked against their actual values.

Details:
-Use 1024*1024 threads
-32x32 threads/block (2-D)
-32x32 blocks/grid (2-D)
-The value of each element of A is 1
-The value of each element of B is 2

Execution Time: 0.189969 seconds
Value of C[453][453] = 2048 which is expected because 1*2*1024 = 2048

# Compilation and Execution
Compilation requires nvcc