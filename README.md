# cuda-matmul-streams
Overlapped CUDA GEMM using 4 streams + cudaMemcpy2DAsync; tiled H2D/compute/D2H with pinned host memory and a tiny RAII device buffer.
