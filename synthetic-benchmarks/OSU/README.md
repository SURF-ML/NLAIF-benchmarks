# OSU-MicroBenchmarks

## Introduction

OSU-MicroBenchmarks is one of the applications that tests the network fabric of a cluster at various levels. Network
latencies and bandwidths are measured for different configurations.

## Rules

1. Application/code that is used for this benchmark must conform to or should be derived from
   https://mvapich.cse.ohio-state.edu/benchmarks/ . The version used should be the latest release version of the
   code base.
2. The benchmark must report results for both HOST and DEVICE, for all operations mentioned below:
   - Point to point
     - latency
     - bandwidth
   - Collective
     - allreduce
     - allgather
     - alltoall
     - bcast
3. The point to point operations must be reported at the following levels or configurations:
   - Intra node
   - Inter node - nodes on the same leaf switch
   - Inter node - nodes on different leaf switches for each additional hop
4. The collective operations must be reported at the following levels or configurations:
   - Full node
   - Nodes on a leaf switch
   - Full partition
5. The benchmarks should be conducted for data type `mpi_float` and the message size range option should be
   `4:67108864`.
6. The number of warmup iterations should be at least 500 and timing iterations should be at least 10000.
7. The underlying software stack used to compile the code base is tenderer's choice but must be reported along with all
   the compilation options used for optimization.
