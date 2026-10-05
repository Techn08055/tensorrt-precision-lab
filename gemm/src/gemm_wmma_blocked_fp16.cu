// gemm_wmma_blocked_fp16.cu
// Step 3 of the GEMM order: tensor cores (WMMA), FP16 in/out, FP32 accumulation,
// C = alpha*A*B + beta*C.
//
// Two WMMA kernels are timed here:
//   wmma      the submitted kernel, unchanged: one 16x16 output fragment per warp,
//             A and B fragments loaded straight from global memory every k step.
//   wmma_blk  the same tensor-core math with the two ideas from the scalar steps
//             applied: A/B tiles staged in shared memory once per block (tiling),
//             and each warp holding a 2x2 grid of accumulator fragments so every
//             loaded fragment feeds two mma_sync calls (register blocking).
// Baselines on the same data: naive, the best scalar kernel (4x4 register
// blocked), and cuBLAS as the ceiling.
//
// Build: nvcc -O3 -arch=sm_89 gemm_wmma_blocked_fp16.cu -o gemm_wmma_blocked_fp16 -lcublas
// Run:   ./gemm_wmma_blocked_fp16 [sizes...]   (default: 256 512 1024 2048 4096; multiples of 64)

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define TILE_SIZE 32
#define WARP_SIZE 32
#define WMMA 16
#define COARSE 4

// --- the submitted WMMA kernel, unchanged -------------------------------------
__global__ void matrix_multiplication_kernel_wmma(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
    if (warpM * WMMA >= M || warpN * WMMA >= N) return;

    wmma::fragment<wmma::matrix_a, WMMA, WMMA, WMMA, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA, WMMA, WMMA, half, wmma::row_major> b_frag;

    wmma::fragment<wmma::accumulator, WMMA, WMMA, WMMA, float> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA, WMMA, WMMA, half> c_frag;

    wmma::fill_fragment(acc_frag, 0.0f);
    int a_row = warpM * WMMA;
    int b_col = warpN * WMMA;

    for (int k = 0; k < K; k += WMMA) {

        wmma::load_matrix_sync(a_frag, A + a_row * K + k, K);
        wmma::load_matrix_sync(b_frag, B + k * N + b_col, N);

        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    wmma::load_matrix_sync(c_frag, C + a_row * N + b_col, N, wmma::mem_row_major);
    for (int i = 0; i < c_frag.num_elements; ++i) {
        c_frag.x[i] = __float2half(alpha * acc_frag.x[i] + beta * __half2float(c_frag.x[i]));
    }

    wmma::store_matrix_sync(C + a_row * N + b_col, c_frag, N, wmma::mem_row_major);

}

// --- blocked WMMA kernel ------------------------------------------------------
// One thread block = 4 warps = one 64x64 tile of C. Each warp owns a 32x32
// quadrant, held as 2x2 accumulator fragments. BK = how many k columns are staged
// in shared memory per outer step. Needs M, N multiples of 64 and K a multiple of BK.
#define BM 64
#define BN 64
#define WARPS_PER_BLOCK 4
#define BLK_THREADS (WARPS_PER_BLOCK * WARP_SIZE)

template <int BK>
__global__ void gemm_wmma_blocked_kernel(const half* A, const half* B, half* C, int M, int N,
                                         int K, float alpha, float beta) {
    __shared__ __align__(32) half As[BM * BK];   // BM rows x BK cols
    __shared__ __align__(32) half Bs[BK * BN];   // BK rows x BN cols

    int tid = threadIdx.x;
    int warp = tid / WARP_SIZE;
    int wr = (warp / 2) * 32;                    // this warp's quadrant inside the tile
    int wc = (warp % 2) * 32;
    int row0 = blockIdx.y * BM;                  // this block's tile inside C
    int col0 = blockIdx.x * BN;

    wmma::fragment<wmma::matrix_a, WMMA, WMMA, WMMA, half, wmma::row_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, WMMA, WMMA, WMMA, half, wmma::row_major> b_frag[2];
    wmma::fragment<wmma::accumulator, WMMA, WMMA, WMMA, float> acc[2][2];
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 2; j++) wmma::fill_fragment(acc[i][j], 0.0f);

    for (int k0 = 0; k0 < K; k0 += BK) {
        // stage the A and B tiles once; all four warps then read them from shared memory
        #pragma unroll
        for (int idx = tid; idx < BM * BK; idx += BLK_THREADS) {
            int r = idx / BK, c = idx % BK;
            As[idx] = A[(row0 + r) * K + k0 + c];
        }
        #pragma unroll
        for (int idx = tid; idx < BK * BN; idx += BLK_THREADS) {
            int r = idx / BN, c = idx % BN;
            Bs[idx] = B[(k0 + r) * N + col0 + c];
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; kk += WMMA) {
            // 2 A fragments + 2 B fragments feed 4 mma_sync calls
            #pragma unroll
            for (int i = 0; i < 2; i++)
                wmma::load_matrix_sync(a_frag[i], As + (wr + i * WMMA) * BK + kk, BK);
            #pragma unroll
            for (int j = 0; j < 2; j++)
                wmma::load_matrix_sync(b_frag[j], Bs + kk * BN + wc + j * WMMA, BN);
            #pragma unroll
            for (int i = 0; i < 2; i++)
                #pragma unroll
                for (int j = 0; j < 2; j++)
                    wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
        }
        __syncthreads();
    }

    // same epilogue as the submitted kernel, once per accumulator fragment
    wmma::fragment<wmma::accumulator, WMMA, WMMA, WMMA, half> c_frag;
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            half* cptr = C + (row0 + wr + i * WMMA) * N + col0 + wc + j * WMMA;
            wmma::load_matrix_sync(c_frag, cptr, N, wmma::mem_row_major);
            for (int e = 0; e < c_frag.num_elements; ++e)
                c_frag.x[e] = __float2half(alpha * acc[i][j].x[e] + beta * __half2float(c_frag.x[e]));
            wmma::store_matrix_sync(cptr, c_frag, N, wmma::mem_row_major);
        }
    }
}

// --- best scalar kernel from the previous step (see gemm_regblock_fp16.cu) ------
template <int R>
__global__ void gemm_regblock_kernel(const half* A, const half* B, half* C, int M, int N, int K,
                                     float alpha, float beta) {
    __shared__ float Ashared[TILE_SIZE][TILE_SIZE];
    __shared__ float Bshared[TILE_SIZE][TILE_SIZE];

    int ty0 = threadIdx.y * R;
    int tx0 = threadIdx.x * R;
    int row0 = blockIdx.y * TILE_SIZE + ty0;
    int col0 = blockIdx.x * TILE_SIZE + tx0;

    float acc[R][R];
    #pragma unroll
    for (int r = 0; r < R; r++)
        #pragma unroll
        for (int c = 0; c < R; c++) acc[r][c] = 0.0f;

    int iters = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int i = 0; i < iters; i++) {
        #pragma unroll
        for (int r = 0; r < R; r++) {
            #pragma unroll
            for (int c = 0; c < R; c++) {
                int arow = row0 + r, acol = i * TILE_SIZE + tx0 + c;
                int brow = i * TILE_SIZE + ty0 + r, bcol = col0 + c;
                Ashared[ty0 + r][tx0 + c] =
                    (arow < M && acol < K) ? __half2float(A[arow * K + acol]) : 0.0f;
                Bshared[ty0 + r][tx0 + c] =
                    (brow < K && bcol < N) ? __half2float(B[brow * N + bcol]) : 0.0f;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            float a[R], b[R];
            #pragma unroll
            for (int r = 0; r < R; r++) a[r] = Ashared[ty0 + r][k];
            #pragma unroll
            for (int c = 0; c < R; c++) b[c] = Bshared[k][tx0 + c];
            #pragma unroll
            for (int r = 0; r < R; r++)
                #pragma unroll
                for (int c = 0; c < R; c++) acc[r][c] += a[r] * b[c];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int r = 0; r < R; r++) {
        #pragma unroll
        for (int c = 0; c < R; c++) {
            int row = row0 + r, col = col0 + c;
            if (row < M && col < N) {
                int idx = row * N + col;
                C[idx] = __float2half(acc[r][c] * alpha + beta * __half2float(C[idx]));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Benchmark driver
// ---------------------------------------------------------------------------

#include <cublas_v2.h>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t e__ = (call);                                               \
        if (e__ != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                    cudaGetErrorString(e__));                                   \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

__global__ void gemm_naive_fp16(const half* A, const half* B, half* C, int M, int N, int K,
                                float alpha, float beta) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += __half2float(A[row * K + k]) * __half2float(B[k * N + col]);
    }
    int idx = row * N + col;
    C[idx] = __float2half(acc * alpha + beta * __half2float(C[idx]));
}

enum Variant { NAIVE, RB4, WMMA_K, WMMA_BLK32, WMMA_BLK64, CUBLAS, NUM_VARIANTS };
static const char* VNAME[] = {"naive", "rb4x4", "wmma", "wmma_blk32", "wmma_blk64", "cublas"};
static cublasHandle_t g_handle;

static void launch(Variant v, const half* dA, const half* dB, half* dC, int M, int N, int K) {
    const float alpha = 1.0f, beta = 0.0f;
    dim3 tile_grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE, 1);
    dim3 blk_grid(N / BN, M / BM, 1);
    switch (v) {
        case NAIVE:
            gemm_naive_fp16<<<tile_grid, dim3(32, 32)>>>(dA, dB, dC, M, N, K, alpha, beta);
            break;
        case RB4:
            gemm_regblock_kernel<4><<<tile_grid, dim3(8, 8)>>>(dA, dB, dC, M, N, K, alpha, beta);
            break;
        case WMMA_K: {
            // identical launch shape to the submitted solve()
            dim3 block(TILE_SIZE, COARSE);
            int dimx = WMMA * COARSE;
            dim3 grid((M + WMMA - 1) / WMMA, (N + dimx - 1) / dimx);
            matrix_multiplication_kernel_wmma<<<grid, block>>>(dA, dB, dC, M, N, K, alpha, beta);
            break;
        }
        case WMMA_BLK32:
            gemm_wmma_blocked_kernel<32><<<blk_grid, dim3(BLK_THREADS)>>>(dA, dB, dC, M, N, K, alpha, beta);
            break;
        case WMMA_BLK64:
            gemm_wmma_blocked_kernel<64><<<blk_grid, dim3(BLK_THREADS)>>>(dA, dB, dC, M, N, K, alpha, beta);
            break;
        default: {
            // cuBLAS is column-major. A row-major MxK array is a column-major KxM
            // array (the transpose), so row-major C = A*B is column-major C^T = B^T*A^T.
            cublasStatus_t st = cublasGemmEx(g_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                                             dB, CUDA_R_16F, N, dA, CUDA_R_16F, K, &beta,
                                             dC, CUDA_R_16F, N, CUBLAS_COMPUTE_32F,
                                             CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            if (st != CUBLAS_STATUS_SUCCESS) {
                fprintf(stderr, "cublasGemmEx failed: %d\n", (int)st);
                exit(1);
            }
        }
    }
}

static float time_variant(Variant v, const half* dA, const half* dB, half* dC, int M, int N,
                          int K, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) launch(v, dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    cudaEvent_t s, e;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&e));
    float total = 0.0f;
    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK(cudaEventRecord(s));
        launch(v, dA, dB, dC, M, N, K);
        CUDA_CHECK(cudaEventRecord(e));
        CUDA_CHECK(cudaEventSynchronize(e));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
        total += ms;
    }
    CUDA_CHECK(cudaGetLastError());
    cudaEventDestroy(s);
    cudaEventDestroy(e);
    return total / iters;
}

static double max_abs_diff(const std::vector<half>& a, const std::vector<half>& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = fabs((double)__half2float(a[i]) - (double)__half2float(b[i]));
        if (!(d <= m)) m = d;  // also catches NaN
    }
    return m;
}

static void run_size(int S, FILE* csv) {
    const int M = S, N = S, K = S;
    std::vector<half> hA((size_t)M * K), hB((size_t)K * N);
    std::vector<half> out[NUM_VARIANTS];
    srand(1234 + S);
    for (auto& v : hA) v = __float2half((float)rand() / RAND_MAX * 2.0f - 1.0f);
    for (auto& v : hB) v = __float2half((float)rand() / RAND_MAX * 2.0f - 1.0f);

    half *dA, *dB, *dC;
    size_t bA = sizeof(half) * hA.size(), bB = sizeof(half) * hB.size();
    size_t bC = sizeof(half) * (size_t)M * N;
    CUDA_CHECK(cudaMalloc(&dA, bA));
    CUDA_CHECK(cudaMalloc(&dB, bB));
    CUDA_CHECK(cudaMalloc(&dC, bC));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), bA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), bB, cudaMemcpyHostToDevice));

    float ms[NUM_VARIANTS];
    double gf[NUM_VARIANTS];
    double flops = 2.0 * M * N * K;
    for (int v = 0; v < NUM_VARIANTS; ++v) {
        CUDA_CHECK(cudaMemset(dC, 0, bC));
        ms[v] = time_variant((Variant)v, dA, dB, dC, M, N, K, 2, 5);
        gf[v] = flops / (ms[v] / 1e3) / 1e9;
        out[v].resize((size_t)M * N);
        CUDA_CHECK(cudaMemcpy(out[v].data(), dC, bC, cudaMemcpyDeviceToHost));
    }

    // CPU fp32 reference on the same half inputs, for sizes that finish quickly
    double ref_err[NUM_VARIANTS];
    for (int v = 0; v < NUM_VARIANTS; ++v) ref_err[v] = -1.0;
    if (S <= 1024) {
        std::vector<float> fA(hA.size()), fB(hB.size()), ref((size_t)M * N, 0.0f);
        for (size_t i = 0; i < hA.size(); ++i) fA[i] = __half2float(hA[i]);
        for (size_t i = 0; i < hB.size(); ++i) fB[i] = __half2float(hB[i]);
        for (int i = 0; i < M; ++i)
            for (int k = 0; k < K; ++k) {
                float a = fA[(size_t)i * K + k];
                const float* brow = &fB[(size_t)k * N];
                float* crow = &ref[(size_t)i * N];
                for (int j = 0; j < N; ++j) crow[j] += a * brow[j];
            }
        for (int v = 0; v < NUM_VARIANTS; ++v) {
            double m = 0.0;
            for (size_t i = 0; i < ref.size(); ++i) {
                double d = fabs((double)__half2float(out[v][i]) - (double)ref[i]);
                if (!(d <= m)) m = d;
            }
            ref_err[v] = m;
        }
    }

    printf("S=%d\n", S);
    for (int v = 0; v < NUM_VARIANTS; ++v) {
        double d_naive = max_abs_diff(out[v], out[NAIVE]);
        double d_wmma = max_abs_diff(out[v], out[WMMA_K]);
        printf("  %-10s %10.4f ms %9.1f GFLOPS  %6.2fx vs naive  %5.2fx vs wmma  %5.1f%% of cublas"
               "  max|d naive| %-8.4g max|d wmma| %-8.4g",
               VNAME[v], ms[v], gf[v], ms[NAIVE] / ms[v], ms[WMMA_K] / ms[v],
               100.0 * ms[CUBLAS] / ms[v], d_naive, d_wmma);
        if (ref_err[v] >= 0) printf(" max|d cpu_fp32| %.4g", ref_err[v]);
        printf("\n");
        fprintf(csv, "%d,%s,%.5f,%.2f,%.3f,%.3f,%.6g,%.6g,%.6g\n", S, VNAME[v], ms[v], gf[v],
                ms[NAIVE] / ms[v], ms[WMMA_K] / ms[v], d_naive, d_wmma,
                ref_err[v] >= 0 ? ref_err[v] : NAN);
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

int main(int argc, char** argv) {
    std::vector<int> sizes;
    for (int i = 1; i < argc; ++i) sizes.push_back(atoi(argv[i]));
    if (sizes.empty()) sizes = {256, 512, 1024, 2048, 4096};
    for (int S : sizes)
        if (S % 64 != 0) {
            fprintf(stderr, "size %d is not a multiple of 64; the blocked WMMA kernel needs that\n", S);
            return 1;
        }

    if (cublasCreate(&g_handle) != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasCreate failed\n");
        return 1;
    }
    FILE* csv = fopen("../results/wmma_blocked_fp16_comparison.csv", "w");
    if (!csv) csv = fopen("wmma_blocked_fp16_comparison.csv", "w");
    fprintf(csv, "S,variant,mean_ms,gflops,speedup_vs_naive,speedup_vs_wmma,max_abs_vs_naive,"
                 "max_abs_vs_wmma,max_abs_vs_cpu_fp32\n");
    for (int S : sizes) run_size(S, csv);
    fclose(csv);
    cublasDestroy(g_handle);
    return 0;
}
