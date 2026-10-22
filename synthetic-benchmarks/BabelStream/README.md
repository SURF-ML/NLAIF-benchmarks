# BabelStream

## Introduction

BabelStream (earlier known as GPU-STREAM) is a STREAM based benchmark which measures the achievable memory bandwidth between VRAM
and the SMs of GPUs.

## Rules

1. Application/code that is used for this benchmark must conform to or should be derived from
   https://github.com/uob-hpc/babelstream .
2. The benchmark must report results for all operations mentioned below:
   - Copy
   - Mul
   - Add
   - Triad
   - Dot
3. The array size chosen must be such that it occupies at the very least 4 times the last level cache to come as
   close to the peak theoretical bandwidth that is possible.
4. This benchmark must be run on each GPU card and along with results for each card, an aggregate and mean must be
   provided for each of the operations across all GPU cards.
5. The precision chosen should be double precision.
6. The underlying software stack used to compile the code base is tenderer's choice but must be reported along with all
   the compilation options used for optimization.
