// gemm_wmma_vec_fp16.cu
// Step 4 of the GEMM order: small wins on top of the blocked WMMA kernel
// (gemm_wmma_blocked_fp16.cu). FP16 in/out, FP32 accumulation, C = alpha*A*B + beta*C.
//
// Three changes, all in gemm_wmma_v2_kernel:
//   1. Vectorized tile loads: 8 halves (one 16-byte uint4) per load instruction
//      instead of one half. Template flag VEC switches it off, so the effect of
//      this change alone can be timed.
//   2. beta == 0 no longer reads C.
//   3. Any M, N, K: edge tiles are zero-padded on load and bounds-checked on store.
//      The epilogue goes through a small shared-memory scratch, which also removes
//      the reliance on float and half accumulator fragments sharing a layout.
//
// Build: nvcc -O3 -arch=sm_89 gemm_wmma_vec_fp16.cu -o gemm_wmma_vec_fp16 -lcublas
// Run:   ./gemm_wmma_vec_fp16 [sizes...]   (default: 256 512 1000 1001 1024 2048 4096)

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

// Block shape shared by both blocked kernels: 4 warps = one 64x64 tile of C, each
// warp a 32x32 quadrant (2x2 accumulator fragments), 32 k-columns staged per step.
#define BM 64
#define BN 64
#define BK 32
#define WARPS_PER_BLOCK 4
#define BLK_THREADS (WARPS_PER_BLOCK * WARP_SIZE)

// --- previous step: blocked WMMA kernel (see gemm_wmma_blocked_fp16.cu) ---------
// Needs M, N multiples of 64 and K a multiple of 32.
__global__ void gemm_wmma_blocked_kernel(const half* A, const half* B, half* C, int M, int N,
                                         int K, float alpha, float beta) {
    __shared__ __align__(32) half As[BM * BK];
    __shared__ __align__(32) half Bs[BK * BN];

    int tid = threadIdx.x;
    int warp = tid / WARP_SIZE;
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

// --- this step: vectorized loads, beta == 0 fix, any M/N/K ----------------------
// Launch with grid (ceil(N/64), ceil(M/64)) and BLK_THREADS threads per block.
template <bool VEC>
__global__ void gemm_wmma_v2_kernel(const half* A, const half* B, half* C, int M, int N, int K,
                                    float alpha, float beta) {
    __shared__ __align__(32) half As[BM * BK];
    __shared__ __align__(32) half Bs[BK * BN];
    __shared__ __align__(32) float Cs[WARPS_PER_BLOCK][WMMA * WMMA];  // epilogue scratch, one per warp

    int tid = threadIdx.x;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    int wr = (warp / 2) * 32;
    int wc = (warp % 2) * 32;
    int row0 = blockIdx.y * BM;
    int col0 = blockIdx.x * BN;

    // A 16-byte load needs a 16-byte aligned address: row stride (K or N halves)
    // must be a multiple of 8, and the tile must not hang over the matrix edge.
    const bool vecA = VEC && (row0 + BM <= M) && (K % 8 == 0);
    const bool vecB = VEC && (col0 + BN <= N) && (N % 8 == 0);
    const half zero = __float2half(0.0f);

    wmma::fragment<wmma::matrix_a, WMMA, WMMA, WMMA, half, wmma::row_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, WMMA, WMMA, WMMA, half, wmma::row_major> b_frag[2];
    wmma::fragment<wmma::accumulator, WMMA, WMMA, WMMA, float> acc[2][2];
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 2; j++) wmma::fill_fragment(acc[i][j], 0.0f);

    for (int k0 = 0; k0 < K; k0 += BK) {
        const bool k_full = (k0 + BK <= K);

        if (vecA && k_full) {
            // 64x32 halves = 256 uint4, 2 per thread (was 16 half loads per thread)
            #pragma unroll
            for (int v = tid; v < BM * BK / 8; v += BLK_THREADS) {
                int r = v / (BK / 8), c = v % (BK / 8);
                ((uint4*)As)[v] = *(const uint4*)(A + (row0 + r) * K + k0 + c * 8);
            }
        } else {
            #pragma unroll
            for (int idx = tid; idx < BM * BK; idx += BLK_THREADS) {
                int gr = row0 + idx / BK, gc = k0 + idx % BK;
                As[idx] = (gr < M && gc < K) ? A[gr * K + gc] : zero;
            }
        }
        if (vecB && k_full) {
            #pragma unroll
            for (int v = tid; v < BK * BN / 8; v += BLK_THREADS) {
                int r = v / (BN / 8), c = v % (BN / 8);
                ((uint4*)Bs)[v] = *(const uint4*)(B + (k0 + r) * N + col0 + c * 8);
            }
        } else {
            #pragma unroll
            for (int idx = tid; idx < BK * BN; idx += BLK_THREADS) {
                int gr = k0 + idx / BN, gc = col0 + idx % BN;
                Bs[idx] = (gr < K && gc < N) ? B[gr * N + gc] : zero;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; kk += WMMA) {
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

    // Epilogue: each accumulator fragment goes to this warp's scratch as a plain
    // row-major 16x16 float block, then the 32 lanes write it out 8 elements each,
    // skipping anything past the matrix edge. C is only read when beta != 0.
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            wmma::store_matrix_sync(Cs[warp], acc[i][j], WMMA, wmma::mem_row_major);
            __syncwarp();
            int frow = row0 + wr + i * WMMA, fcol = col0 + wc + j * WMMA;
            #pragma unroll
            for (int e = lane; e < WMMA * WMMA; e += WARP_SIZE) {
                int gr = frow + e / WMMA, gc = fcol + e % WMMA;
                if (gr < M && gc < N) {
                    int idx = gr * N + gc;
                    float v = alpha * Cs[warp][e];
                    if (beta != 0.0f) v += beta * __half2float(C[idx]);
                    C[idx] = __float2half(v);
                }
            }
            __syncwarp();
        }
    }
}

// A, B, C are device pointers. Works for any M, N, K.
extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha,
                      float beta) {
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, 1);
    gemm_wmma_v2_kernel<true><<<grid, dim3(BLK_THREADS)>>>(A, B, C, M, N, K, alpha, beta);
    cudaDeviceSynchronize();
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

// v2_scalar = v2 kernel with VEC off (edge handling + new epilogue only);
// v2_vec = the full step-4 kernel.
enum Variant { NAIVE, WMMA_K, WMMA_BLK, V2_SCALAR, V2_VEC, CUBLAS, NUM_VARIANTS };
static const char* VNAME[] = {"naive", "wmma", "wmma_blk", "v2_scalar", "v2_vec", "cublas"};
static cublasHandle_t g_handle;

// the two earlier WMMA kernels have no edge handling
static bool supported(Variant v, int S) {
    if (v == WMMA_K) return S % 16 == 0;
    if (v == WMMA_BLK) return S % 64 == 0;
    return true;
}

static void launch(Variant v, const half* dA, const half* dB, half* dC, int M, int N, int K) {
    const float alpha = 1.0f, beta = 0.0f;
    dim3 blk_grid((N + BN - 1) / BN, (M + BM - 1) / BM, 1);
    switch (v) {
        case NAIVE: {
            dim3 grid((N + 31) / 32, (M + 31) / 32, 1);
            gemm_naive_fp16<<<grid, dim3(32, 32)>>>(dA, dB, dC, M, N, K, alpha, beta);
            break;
        }
        case WMMA_K: {
            dim3 block(TILE_SIZE, COARSE);
            int dimx = WMMA * COARSE;
            dim3 grid((M + WMMA - 1) / WMMA, (N + dimx - 1) / dimx);
            matrix_multiplication_kernel_wmma<<<grid, block>>>(dA, dB, dC, M, N, K, alpha, beta);
            break;
        }
        case WMMA_BLK:
            gemm_wmma_blocked_kernel<<<blk_grid, dim3(BLK_THREADS)>>>(dA, dB, dC, M, N, K, alpha, beta);
            break;
        case V2_SCALAR:
            gemm_wmma_v2_kernel<false><<<blk_grid, dim3(BLK_THREADS)>>>(dA, dB, dC, M, N, K, alpha, beta);
            break;
        case V2_VEC:
            gemm_wmma_v2_kernel<true><<<blk_grid, dim3(BLK_THREADS)>>>(dA, dB, dC, M, N, K, alpha, beta);
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
        if (!supported((Variant)v, S)) continue;
        CUDA_CHECK(cudaMemset(dC, 0, bC));
        ms[v] = time_variant((Variant)v, dA, dB, dC, M, N, K, 2, 5);
        gf[v] = flops / (ms[v] / 1e3) / 1e9;
        out[v].resize((size_t)M * N);
        CUDA_CHECK(cudaMemcpy(out[v].data(), dC, bC, cudaMemcpyDeviceToHost));
    }

    // CPU fp32 reference on the same half inputs, for sizes that finish quickly
    std::vector<float> ref;
    if (S <= 1024) {
        std::vector<float> fA(hA.size()), fB(hB.size());
        ref.assign((size_t)M * N, 0.0f);
        for (size_t i = 0; i < hA.size(); ++i) fA[i] = __half2float(hA[i]);
        for (size_t i = 0; i < hB.size(); ++i) fB[i] = __half2float(hB[i]);
        for (int i = 0; i < M; ++i)
            for (int k = 0; k < K; ++k) {
                float a = fA[(size_t)i * K + k];
                const float* brow = &fB[(size_t)k * N];
                float* crow = &ref[(size_t)i * N];
                for (int j = 0; j < N; ++j) crow[j] += a * brow[j];
            }
    }

    printf("S=%d\n", S);
    for (int v = 0; v < NUM_VARIANTS; ++v) {
        if (!supported((Variant)v, S)) {
            printf("  %-10s (not supported at this size)\n", VNAME[v]);
            continue;
        }
        double d_naive = max_abs_diff(out[v], out[NAIVE]);
        double d_blk = supported(WMMA_BLK, S) ? max_abs_diff(out[v], out[WMMA_BLK]) : NAN;
        double d_cublas = max_abs_diff(out[v], out[CUBLAS]);
        double d_ref = NAN;
        if (!ref.empty()) {
            d_ref = 0.0;
            for (size_t i = 0; i < ref.size(); ++i) {
                double d = fabs((double)__half2float(out[v][i]) - (double)ref[i]);
                if (!(d <= d_ref)) d_ref = d;
            }
        }
        double vs_blk = supported(WMMA_BLK, S) ? ms[WMMA_BLK] / ms[v] : NAN;
        printf("  %-10s %10.4f ms %9.1f GFLOPS  %6.2fx vs naive  %5.2fx vs wmma_blk  %5.1f%% of cublas"
               "  max|d naive| %-8.4g |d wmma_blk| %-8.4g |d cublas| %-8.4g |d cpu_fp32| %.4g\n",
               VNAME[v], ms[v], gf[v], ms[NAIVE] / ms[v], vs_blk, 100.0 * ms[CUBLAS] / ms[v],
               d_naive, d_blk, d_cublas, d_ref);
        fprintf(csv, "%d,%s,%.5f,%.2f,%.3f,%.3f,%.3f,%.6g,%.6g,%.6g,%.6g\n", S, VNAME[v], ms[v],
                gf[v], ms[NAIVE] / ms[v], vs_blk, ms[CUBLAS] / ms[v], d_naive, d_blk, d_cublas,
                d_ref);
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

// Fill C with NaN, run solve() with alpha=1, beta=0, count NaNs that come out.
// S is deliberately not a multiple of anything, so the edge path is exercised too.
static void beta_zero_nan_check() {
    const int S = 100;
    std::vector<half> hA(S * S), hB(S * S), hC(S * S);
    for (auto& v : hA) v = __float2half(0.5f);
    for (auto& v : hB) v = __float2half(0.5f);
    half *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, sizeof(half) * S * S));
    CUDA_CHECK(cudaMalloc(&dB, sizeof(half) * S * S));
    CUDA_CHECK(cudaMalloc(&dC, sizeof(half) * S * S));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), sizeof(half) * S * S, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), sizeof(half) * S * S, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dC, 0xFF, sizeof(half) * S * S));  // 0xFFFF = half NaN
    solve(dA, dB, dC, S, S, S, 1.0f, 0.0f);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, sizeof(half) * S * S, cudaMemcpyDeviceToHost));
    int nans = 0, wrong = 0;
    for (auto& v : hC) {
        float f = __half2float(v);
        if (isnan(f)) ++nans;
        else if (f != 0.25f * S) ++wrong;
    }
    printf("\nbeta=0 with NaN-filled C, S=%d: %d NaN, %d wrong, out of %d (correct value %.1f)\n",
           S, nans, wrong, S * S, 0.25f * S);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

int main(int argc, char** argv) {
    std::vector<int> sizes;
    for (int i = 1; i < argc; ++i) sizes.push_back(atoi(argv[i]));
    if (sizes.empty()) sizes = {256, 512, 1000, 1001, 1024, 2048, 4096};

    if (cublasCreate(&g_handle) != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasCreate failed\n");
        return 1;
    }
    FILE* csv = fopen("../results/wmma_vec_fp16_comparison.csv", "w");
    if (!csv) csv = fopen("wmma_vec_fp16_comparison.csv", "w");
    fprintf(csv, "S,variant,mean_ms,gflops,speedup_vs_naive,speedup_vs_wmma_blk,fraction_of_cublas,"
                 "max_abs_vs_naive,max_abs_vs_wmma_blk,max_abs_vs_cublas,max_abs_vs_cpu_fp32\n");
    for (int S : sizes) run_size(S, csv);
    fclose(csv);

    beta_zero_nan_check();
    cublasDestroy(g_handle);
    return 0;
}
