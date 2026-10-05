// ablate_wmma_vec.cu
// Where does the step-4 WMMA kernel (src/gemm_wmma_vec_fp16.cu) spend its time?
// Nsight Compute needs admin access to GPU counters on this machine, so this
// answers the question by removing one part of the kernel at a time and timing
// what is left. Modes 1-3 compute wrong results on purpose; only time matters.
//
//   mode 0  full kernel
//   mode 1  global->shared tile loads only on the first k step
//           (tensor-core work + shared-memory fragment loads remain)
//   mode 2  tile loads every step, but no fragment loads and no mma_sync
//           (global memory traffic + staging remain)
//   mode 3  tiles staged once, fragments loaded once, mma_sync repeated
//           (tensor-core work alone, for this 4-warp / 2x2-fragment shape)
//
// Build: nvcc -O3 -arch=sm_89 ablate_wmma_vec.cu -o ablate_wmma_vec
// Run:   ./ablate_wmma_vec [sizes...]     (default: 1024 2048 4096; multiples of 64)

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <vector>

using namespace nvcuda;

#define WARP_SIZE 32
#define WMMA 16
#define BM 64
#define BN 64
#define BK 32
#define WARPS_PER_BLOCK 4
#define BLK_THREADS (WARPS_PER_BLOCK * WARP_SIZE)

template <int MODE>
__global__ void ablate_kernel(const half* A, const half* B, half* C, int M, int N, int K,
                              float alpha, float beta) {
    __shared__ __align__(32) half As[BM * BK];
    __shared__ __align__(32) half Bs[BK * BN];
    __shared__ __align__(32) float Cs[WARPS_PER_BLOCK][WMMA * WMMA];

    int tid = threadIdx.x;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    int wr = (warp / 2) * 32;
    int wc = (warp % 2) * 32;
    int row0 = blockIdx.y * BM;
    int col0 = blockIdx.x * BN;

    wmma::fragment<wmma::matrix_a, WMMA, WMMA, WMMA, half, wmma::row_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, WMMA, WMMA, WMMA, half, wmma::row_major> b_frag[2];
    wmma::fragment<wmma::accumulator, WMMA, WMMA, WMMA, float> acc[2][2];
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 2; j++) wmma::fill_fragment(acc[i][j], 0.0f);

    for (int k0 = 0; k0 < K; k0 += BK) {
        if (MODE == 0 || MODE == 2 || k0 == 0) {
            #pragma unroll
            for (int v = tid; v < BM * BK / 8; v += BLK_THREADS) {
                int r = v / (BK / 8), c = v % (BK / 8);
                ((uint4*)As)[v] = *(const uint4*)(A + (row0 + r) * K + k0 + c * 8);
            }
            #pragma unroll
            for (int v = tid; v < BK * BN / 8; v += BLK_THREADS) {
                int r = v / (BN / 8), c = v % (BN / 8);
                ((uint4*)Bs)[v] = *(const uint4*)(B + (k0 + r) * N + col0 + c * 8);
            }
        }
        __syncthreads();

        if (MODE != 2) {
            #pragma unroll
            for (int kk = 0; kk < BK; kk += WMMA) {
                if (MODE != 3 || k0 == 0) {
                    #pragma unroll
                    for (int i = 0; i < 2; i++)
                        wmma::load_matrix_sync(a_frag[i], As + (wr + i * WMMA) * BK + kk, BK);
                    #pragma unroll
                    for (int j = 0; j < 2; j++)
                        wmma::load_matrix_sync(b_frag[j], Bs + kk * BN + wc + j * WMMA, BN);
                }
                #pragma unroll
                for (int i = 0; i < 2; i++)
                    #pragma unroll
                    for (int j = 0; j < 2; j++)
                        wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
            }
        }
        __syncthreads();
    }

    if (MODE == 2) {
        // read the staged tiles once so the staging stores cannot be discarded
        wmma::load_matrix_sync(a_frag[0], As + wr * BK, BK);
        wmma::load_matrix_sync(b_frag[0], Bs + wc, BN);
        wmma::mma_sync(acc[0][0], a_frag[0], b_frag[0], acc[0][0]);
    }

    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            wmma::store_matrix_sync(Cs[warp], acc[i][j], WMMA, wmma::mem_row_major);
            __syncwarp();
            int frow = row0 + wr + i * WMMA, fcol = col0 + wc + j * WMMA;
            #pragma unroll
            for (int e = lane; e < WMMA * WMMA; e += WARP_SIZE) {
                int idx = (frow + e / WMMA) * N + fcol + e % WMMA;
                float v = alpha * Cs[warp][e];
                if (beta != 0.0f) v += beta * __half2float(C[idx]);
                C[idx] = __float2half(v);
            }
            __syncwarp();
        }
    }
}

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t e__ = (call);                                               \
        if (e__ != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                    cudaGetErrorString(e__));                                   \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

static void launch(int mode, const half* dA, const half* dB, half* dC, int S) {
    dim3 grid(S / BN, S / BM, 1), block(BLK_THREADS);
    switch (mode) {
        case 0: ablate_kernel<0><<<grid, block>>>(dA, dB, dC, S, S, S, 1.0f, 0.0f); break;
        case 1: ablate_kernel<1><<<grid, block>>>(dA, dB, dC, S, S, S, 1.0f, 0.0f); break;
        case 2: ablate_kernel<2><<<grid, block>>>(dA, dB, dC, S, S, S, 1.0f, 0.0f); break;
        default: ablate_kernel<3><<<grid, block>>>(dA, dB, dC, S, S, S, 1.0f, 0.0f); break;
    }
}

int main(int argc, char** argv) {
    std::vector<int> sizes;
    for (int i = 1; i < argc; ++i) sizes.push_back(atoi(argv[i]));
    if (sizes.empty()) sizes = {1024, 2048, 4096};
    static const char* names[] = {"0 full", "1 no global loads", "2 loads only, no mma",
                                  "3 mma only"};

    for (int S : sizes) {
        if (S % 64 != 0) { fprintf(stderr, "size must be a multiple of 64\n"); return 1; }
        size_t n = (size_t)S * S;
        std::vector<half> h(n);
        srand(1234 + S);
        for (auto& v : h) v = __float2half((float)rand() / RAND_MAX * 2.0f - 1.0f);
        half *dA, *dB, *dC;
        CUDA_CHECK(cudaMalloc(&dA, n * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&dB, n * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&dC, n * sizeof(half)));
        CUDA_CHECK(cudaMemcpy(dA, h.data(), n * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, h.data(), n * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(dC, 0, n * sizeof(half)));

        printf("S=%d\n", S);
        float full_ms = 0.0f;
        for (int mode = 0; mode < 4; ++mode) {
            for (int i = 0; i < 2; ++i) launch(mode, dA, dB, dC, S);
            CUDA_CHECK(cudaDeviceSynchronize());
            cudaEvent_t s, e;
            CUDA_CHECK(cudaEventCreate(&s));
            CUDA_CHECK(cudaEventCreate(&e));
            float total = 0.0f;
            const int iters = 5;
            for (int i = 0; i < iters; ++i) {
                CUDA_CHECK(cudaEventRecord(s));
                launch(mode, dA, dB, dC, S);
                CUDA_CHECK(cudaEventRecord(e));
                CUDA_CHECK(cudaEventSynchronize(e));
                float ms;
                CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
                total += ms;
            }
            CUDA_CHECK(cudaGetLastError());
            float ms = total / iters;
            if (mode == 0) full_ms = ms;
            // GFLOPS as if the full 2*S^3 work had been done in this time
            printf("  mode %-22s %9.4f ms  %5.1f%% of full  (%8.1f GFLOPS-equivalent)\n",
                   names[mode], ms, 100.0f * ms / full_ms, 2.0 * S * S * S / (ms / 1e3) / 1e9);
            cudaEventDestroy(s);
            cudaEventDestroy(e);
        }
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
    }
    return 0;
}
