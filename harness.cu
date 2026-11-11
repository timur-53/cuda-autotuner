// Fixed benchmark harness: correctness check vs. trusted reference + cudaEvent timing.
// Build: nvcc -O3 harness.cu kernels/<variant>.cu -o bench
// Prints one JSON line to stdout.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

void launch(const float* A, const float* B, float* C, int N);

__global__ void ref_matmul(const float* A, const float* B, float* C, int N) {
    int r = blockIdx.y * blockDim.y + threadIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < N && c < N) {
        double acc = 0.0;
        for (int k = 0; k < N; ++k) acc += (double)A[r * N + k] * B[k * N + c];
        C[r * N + c] = (float)acc;
    }
}

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("{\"ok\":false,\"error\":\"%s\"}\n", cudaGetErrorString(e)); return 1; } } while (0)

int main(int argc, char** argv) {
    int N = argc > 1 ? atoi(argv[1]) : 1024;
    size_t bytes = (size_t)N * N * sizeof(float);
    std::vector<float> hA(N * N), hB(N * N), hC(N * N), hR(N * N);
    srand(0);
    for (auto& x : hA) x = rand() / (float)RAND_MAX - 0.5f;
    for (auto& x : hB) x = rand() / (float)RAND_MAX - 0.5f;

    float *dA, *dB, *dC, *dR;
    CK(cudaMalloc(&dA, bytes)); CK(cudaMalloc(&dB, bytes));
    CK(cudaMalloc(&dC, bytes)); CK(cudaMalloc(&dR, bytes));
    CK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));
    CK(cudaMemset(dC, 0, bytes));

    dim3 b(16, 16), g((N + 15) / 16, (N + 15) / 16);
    ref_matmul<<<g, b>>>(dA, dB, dR, N);
    CK(cudaGetLastError());

    // Correctness
    launch(dA, dB, dC, N);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hR.data(), dR, bytes, cudaMemcpyDeviceToHost));
    double max_err = 0;
    for (int i = 0; i < N * N; ++i) max_err = fmax(max_err, fabs((double)hC[i] - hR[i]));
    if (!(max_err < 1e-2)) {
        printf("{\"ok\":false,\"error\":\"incorrect result\",\"max_err\":%g}\n", max_err);
        return 0;
    }

    // Timing with cudaEvent_t
    const int warmup = 5, iters = 20;
    for (int i = 0; i < warmup; ++i) launch(dA, dB, dC, N);
    CK(cudaDeviceSynchronize());
    cudaEvent_t s, e;
    CK(cudaEventCreate(&s)); CK(cudaEventCreate(&e));
    CK(cudaEventRecord(s));
    for (int i = 0; i < iters; ++i) launch(dA, dB, dC, N);
    CK(cudaEventRecord(e)); CK(cudaEventSynchronize(e));
    float ms; CK(cudaEventElapsedTime(&ms, s, e));
    ms /= iters;
    double gflops = 2.0 * N * N * (double)N / (ms * 1e6);
    printf("{\"ok\":true,\"time_ms\":%.4f,\"gflops\":%.2f,\"max_err\":%g,\"N\":%d}\n", ms, gflops, max_err, N);
    return 0;
}
