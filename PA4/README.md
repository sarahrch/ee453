Sarah Chow - 4091216117
EE453 PA4

# Compilation and Execution
Compilation requires nvcc
**make** will produce problem1.exe
**problem1** will run problem1.exe

# Program Description
This program uses CUDA programming to parallel compute the "distance" between a value and four mean cluster values for the k-means algorithm.
Input: input.raw
Output: output.raw of data after k-means classification
    - output_figure.png included

Details:
- Using 800*800 threads (1250 blocks of 512 threads each)
- Initial cluster values are: 0, 85, 170, and 255
- The algorithm runs a pre-defined 30 iterations, rather than basing further iterations on a check for convergence.

Results: 
Execution Time: 0.124031 seconds using Google Colab GPU

The k-means values do converge after 11 iterations:
    Means for iteration 0: c1=8 c2= 76 c3= 167 c4= 249
    Means for iteration 1: c1=8 c2= 74 c3= 162 c4= 248
    Means for iteration 2: c1=8 c2= 72 c3= 159 c4= 247
    Means for iteration 3: c1=7 c2= 70 c3= 156 c4= 247
    Means for iteration 4: c1=7 c2= 68 c3= 154 c4= 246
    Means for iteration 5: c1=7 c2= 66 c3= 152 c4= 246
    Means for iteration 6: c1=7 c2= 65 c3= 151 c4= 246
    Means for iteration 7: c1=7 c2= 65 c3= 150 c4= 245
    Means for iteration 8: c1=7 c2= 64 c3= 149 c4= 245
    Means for iteration 9: c1=6 c2= 63 c3= 148 c4= 245
    Means for iteration 10: c1=6 c2= 62 c3= 147 c4= 245
    Means for iteration 11: c1=6 c2= 62 c3= 146 c4= 245
    Means for iteration 12: c1=6 c2= 62 c3= 146 c4= 245
    Means for iteration 13: c1=6 c2= 62 c3= 146 c4= 245
    Means for iteration 14: c1=6 c2= 62 c3= 146 c4= 245
    Means for iteration 15: c1=6 c2= 62 c3= 146 c4= 245
    Means for iteration 16: c1=6 c2= 62 c3= 146 c4= 245
    Means for iteration 17: c1=6 c2= 62 c3= 146 c4= 245
    Means for iteration 18: c1=6 c2= 62 c3= 146 c4= 245
    Means for iteration 19: c1=6 c2= 62 c3= 146 c4= 245
    Means for iteration 20: c1=6 c2= 62 c3= 146 c4= 245
    Means for iteration 21: c1=6 c2= 62 c3= 146 c4= 245
    Means for iteration 22: c1=6 c2= 62 c3= 146 c4= 245
    Means for iteration 23: c1=6 c2= 62 c3= 146 c4= 245
    Means for iteration 24: c1=6 c2= 62 c3= 146 c4= 245
    Means for iteration 25: c1=6 c2= 62 c3= 146 c4= 245
    Means for iteration 26: c1=6 c2= 62 c3= 146 c4= 245
    Means for iteration 27: c1=6 c2= 62 c3= 146 c4= 245
    Means for iteration 28: c1=6 c2= 62 c3= 146 c4= 245
    Means for iteration 29: c1=6 c2= 62 c3= 146 c4= 245

