# GEMM

## Introduction

The General Matrix Matrix Multiplication benchmark is very standard especially because it is the core hardware based
operation that a GPU is designed for, particularly for AI applications. The core operation performed during these
computations is mathematically defined below:

<!-- prettier-ignore -->
$$ D = \alpha A_{ij}B_{jk} + \beta C $$

## Rules

1. The tenderer is free to choose any GPU vendor based BLAS library or an open-source implementation that conforms to
   the GPU vendor BLAS library. The version used should be the latest release version of the code base and should be
   reproducible by the tenderee.
2. The intent is to attain the maximum FLOPS/s or TOPS/s close to theoretical hardware performance reported by the GPU
   vendor.
3. Input matrices should use a Gaussian distribution with a mean of 0 and standard deviation of 4. Negative values must
   be included as elements in the input matrices.
4. GEMM operations must be performed both with and without sparsity with following input matrix sizes:

   | Operation Type                       | size(M) | size(N) | size(K) | Batch size |
   | ------------------------------------ | ------- | ------- | ------- | ---------- |
   | General GEMM (both dense and sparse) | 8192    | 8192    | 8192    | 1          |
   | Batched GEMM (both dense and sparse) | 1024    | 1024    | 1024    | 64         |

   where dimensions of $A = M\times K$, $B=K\times N$ and $C=M\times N$.

5. The performance in FLOPS/s or TOPS/s must be reported for each precision matrix shown below and appropriate
   accumulator may be chosen, either BF16 or FP32 or INT8 whichever is applicable.

   | D     | A     | B     | C     |
   | ----- | ----- | ----- | ----- |
   | MXFP4 | MXFP4 | MXFP4 | MXFP4 |
   | BF16  | MXFP4 | MXFP4 | MXFP4 |
   | FP8   | FP8   | FP8   | FP8   |
   | FP16  | FP8   | FP8   | FP8   |
   | FP16  | FP16  | FP16  | FP16  |
   | BF16  | BF16  | BF16  | BF16  |
   | FP32  | BF16  | BF16  | FP32  |
   | TF32  | TF32  | TF32  | TF32  |
   | INT8  | INT8  | INT8  | INT8  |

6. This benchmark must be run for each GPU card and an aggregate performance must be reported for all GPU cards within
   the system for each category and precision.
7. The software stack used to compile the library or the code base used should be reported in detail to the tenderee
   along with all the compilation options used for optimization.
