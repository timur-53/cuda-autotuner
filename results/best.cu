#include <cuda_runtime.h>

#define BM 128
#define BN 64
#define BK 16
#define APAD 4

template <bool FAST>
__global__ void __launch_bounds__(256)
matmul_reg(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N) {
    __shared__ __align__(16) float As[2][BK][BM + APAD];
    __shared__ __align__(16) float Bs[2][BK][BN];

    const int tid = threadIdx.x;
    const int ty = tid >> 4;   // 0..15
    const int tx = tid & 15;   // 0..15
    const int rowBase = blockIdx.y * BM;
    const int colBase = blockIdx.x * BN;

    float acc[8][4];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;

    const int numTiles = (N + BK - 1) / BK;

    float4 pa[2], pb;       // FAST
    float sa[8], sb[4];     // slow

    auto gload = [&](int k0) {
        if (FAST) {
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                int f = tid + i * 256;
                int r = f >> 2, kc = (f & 3) * 4;
                pa[i] = *reinterpret_cast<const float4*>(&A[(size_t)(rowBase + r) * N + k0 + kc]);
            }
            int br = tid >> 4, bc = (tid & 15) * 4;
            pb = *reinterpret_cast<const float4*>(&B[(size_t)(k0 + br) * N + colBase + bc]);
        } else {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                int e = tid + i * 256;
                int r = e >> 4, kk = e & 15;
                int gr = rowBase + r, gk = k0 + kk;
                sa[i] = (gr < N && gk < N) ? A[(size_t)gr * N + gk] : 0.f;
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                int e = tid + i * 256;
                int bk = e >> 6, c = e & 63;
                int gbk = k0 + bk, gc = colBase + c;
                sb[i] = (gbk < N && gc < N) ? B[(size_t)gbk * N + gc] : 0.f;
            }
        }
    };
    auto sstore = [&](int buf) {
        if (FAST) {
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                int f = tid + i * 256;
                int r = f >> 2, kc = (f & 3) * 4;
                As[buf][kc + 0][r] = pa[i].x; As[buf][kc + 1][r] = pa[i].y;
                As[buf][kc + 2][r] = pa[i].z; As[buf][kc + 3][r] = pa[i].w;
            }
            int br = tid >> 4, bc = (tid & 15) * 4;
            *reinterpret_cast<float4*>(&Bs[buf][br][bc]) = pb;
        } else {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                int e = tid + i * 256;
                int r = e >> 4, kk = e & 15;
                As[buf][kk][r] = sa[i];
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                int e = tid + i * 256;
                int bk = e >> 6, c = e & 63;
                Bs[buf][bk][c] = sb[i];
            }
        }
    };

    gload(0);
    sstore(0);
    __syncthreads();

    for (int t = 0; t < numTiles; ++t) {
        const int cur = t & 1;
        const bool has_next = (t + 1) < numTiles;
        if (has_next) gload((t + 1) * BK);

#pragma unroll
        for (int k = 0; k < BK; ++k) {
            float4 a0 = *reinterpret_cast<const float4*>(&As[cur][k][ty * 4]);
            float4 a1 = *reinterpret_cast<const float4*>(&As[cur][k][64 + ty * 4]);
            float4 b0 = *reinterpret_cast<const float4*>(&Bs[cur][k][tx * 4]);
            float a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            float b[4] = {b0.x, b0.y, b0.z, b0.w};
#pragma unroll
            for (int i = 0; i < 8; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }

        if (has_next) sstore(cur ^ 1);
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        int r = rowBase + (i < 4 ? ty * 4 + i : 64 + ty * 4 + (i - 4));
        int c = colBase + tx * 4;
        if (FAST) {
            *reinterpret_cast<float4*>(&C[(size_t)r * N + c]) =
                make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
        } else {
            if (r < N) {
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    if (c + j < N) C[(size_t)r * N + c + j] = acc[i][j];
            }
        }
    }
}

void launch(const float* A, const float* B, float* C, int N) {
    dim3 block(256);
    dim3 grid((N + BN - 1) / BN, (N + BM - 1) / BM);
    if (N % 128 == 0)
        matmul_reg<true><<<grid, block>>>(A, B, C, N);
    else
        matmul_reg<false><<<grid, block>>>(A, B, C, N);
}
