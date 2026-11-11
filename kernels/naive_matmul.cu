// Baseline kernel. The agent rewrites this file; the harness (harness.cu) stays fixed.
// Contract: implement launch(A, B, C, N) computing C = A * B for row-major NxN floats.
#include <cuda_runtime.h>

__global__ void matmul_naive(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float acc = 0.f;
        for (int k = 0; k < N; ++k) acc += A[row * N + k] * B[k * N + col];
        C[row * N + col] = acc;
    }
}

void launch(const float* A, const float* B, float* C, int N) {
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    matmul_naive<<<grid, block>>>(A, B, C, N);
}
