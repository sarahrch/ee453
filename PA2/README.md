Sarah Chow - 4091216117
EE453 PA 2

# Description:
This program contains two hard-coded arrays (representing 1x5 and 5x1 vectors) to be multiplied together.
Both the size of the vectors and their contents may be modified by editing the code.
10 threads and 5 pipes are used to execute this multiplication according to the assignment guidelines (but this process could be optimized from this implementation).

# Compilation and Execution
Compile using: make
Run using: ./matrixmult

# Part 1:
Multiplying A = [1, 2, 3, 4, 5] and B = [6, 7, 8, 9, 10]

Final multiplication result: 130
Execution Time: ~0.001282 seconds

# Part 2:

1. inscount.out: Count 2431474

2. itrace.out: First 10 lines

0x7f2ec585a100
0x7f2ec585a103
0x7f2ec585adf0
0x7f2ec585adf4
0x7f2ec585adf5
0x7f2ec585adf8
0x7f2ec585adfa
0x7f2ec585adfd
0x7f2ec585adff
0x7f2ec585ae01


3. pinatrace.out: First 10 lines

0x7fac0e214103: W 0x7fffb55e7258
0x7fac0e214df4: W 0x7fffb55e7250
0x7fac0e214df8: W 0x7fffb55e7248
0x7fac0e214dfd: W 0x7fffb55e7240
0x7fac0e214dff: W 0x7fffb55e7238
0x7fac0e214e01: W 0x7fffb55e7230
0x7fac0e214e03: W 0x7fffb55e7228
0x7fac0e214e18: W 0x7fac0e2405e0
0x7fac0e214e1f: R 0x7fac0e240e68
0x7fac0e214e29: R 0x7fac0e241000
