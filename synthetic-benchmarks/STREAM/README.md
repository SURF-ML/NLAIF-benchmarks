# STREAM

## Introduction

STREAM benchmark is a standard one for any high performance cluster measuring CPU DRAM memory bandwidth for different
operations.

## Rules

1. Application/code that is used for this benchmark must conform to or should be derived from
   https://www.cs.virginia.edu/stream/ref.html . The version used should be the latest release version of the code base.
2. The benchmark must report results for all four major operations:
   - COPY
   - SCALE
   - ADD
   - TRIAD
3. The array size chosen must be such that it occupies at the very least 4 times the last level cache.
4. This benchmark must be run on each node and along with results for each node, an aggregate and mean must be provided
   for each of the operations.
5. For multi-processor runs, MPI can be chosen if OpenMP support is not available.
6. The underlying software stack used to compile the code base is tenderer's choice but must be reported along with all
   the options used for compilation.
