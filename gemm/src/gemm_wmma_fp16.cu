// gemm_wmma_fp16.cu
// Tensor-core (WMMA) GEMM, FP16 in/out, FP32 accumulation, C = alpha*A*B + beta*C.
// Both kernels and solve() below are the submitted version, unchanged.
// The driver at the bottom times four things on the same data: a naive kernel,
// the shared-memory tiled kernel, the WMMA kernel, and cuBLAS as the ceiling.
//
// Build: nvcc -O3 -arch=sm_89 gemm_wmma_fp16.cu -o gemm_wmma_fp16 -lcublas
// Run:   ./gemm_wmma_fp16 [sizes...]     (default: 256 512 1024 2048 4096; multiples of 16)

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


__global__ void matrix_multiplication_kernel_simp(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {

    __shared__ float Ashared[TILE_SIZE][TILE_SIZE];
    __shared__ float Bshared[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float Pvalue1 = 0;
    int iters = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int i = 0; i < iters; i++) {
        int acol = TILE_SIZE * i + tx;
        int brow = i * TILE_SIZE + ty;

        if (row < M && acol < K) {
            Ashared[ty][tx] = __half2float(A[K * row + acol]);
        } else {
            Ashared[ty][tx] = 0.0f;
        }
        if (brow < K && col < N) {
            Bshared[ty][tx] = __half2float(B[brow * N + col]);
        } else {
            Bshared[ty][tx] = 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            Pvalue1 += Ashared[ty][k] * Bshared[k][tx];
        }
        __syncthreads();
    }
    int idx = row * N + col;

    if (row < M && col < N) {
        half out0 = __float2half(Pvalue1 * alpha + beta * __half2float(C[idx]));
        C[idx] = out0;
    }
}


// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C"
void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    bool use_wmma = (M % WMMA == 0) && (N % WMMA == 0) && (K % WMMA == 0);
    if (use_wmma) {
        dim3 dimBlock(TILE_SIZE, COARSE);
        int dimx = WMMA * COARSE;
        dim3 dimGrid(
            (M + WMMA - 1) / WMMA,
            (N + dimx - 1) / dimx
        );
        matrix_multiplication_kernel_wmma<<<dimGrid, dimBlock>>>(A, B, C, M, N, K, alpha, beta);
    } else {
        int dimRows = (M + TILE_SIZE - 1) / TILE_SIZE;
        int dimCols = (N + TILE_SIZE - 1) / TILE_SIZE;
        dim3 dimGrid( dimCols, dimRows, 1);
        dim3 dimBlock( TILE_SIZE, TILE_SIZE, 1);
        matrix_multiplication_kernel_simp<<<dimGrid, dimBlock>>>(A, B, C, M, N, K, alpha, beta);

   }
    cudaDeviceSynchronize();

}

// ---------------------------------------------------------------------------
// Benchmark driver (not part of the submission)
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

enum Variant { NAIVE, TILED, WMMA_K, CUBLAS, NUM_VARIANTS };
static const char* VNAME[] = {"naive", "tiled", "wmma", "cublas"};
static cublasHandle_t g_handle;

static void launch(Variant v, const half* dA, const half* dB, half* dC, int M, int N, int K) {
    const float alpha = 1.0f, beta = 0.0f;
    if (v == NAIVE || v == TILED) {
        dim3 block(TILE_SIZE, TILE_SIZE, 1);
        dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE, 1);
        if (v == NAIVE) gemm_naive_fp16<<<grid, block>>>(dA, dB, dC, M, N, K, alpha, beta);
        else matrix_multiplication_kernel_simp<<<grid, block>>>(dA, dB, dC, M, N, K, alpha, beta);
    } else if (v == WMMA_K) {
        // identical launch shape to solve()
        dim3 block(TILE_SIZE, COARSE);
        int dimx = WMMA * COARSE;
        dim3 grid((M + WMMA - 1) / WMMA, (N + dimx - 1) / dimx);
        matrix_multiplication_kernel_wmma<<<grid, block>>>(dA, dB, dC, M, N, K, alpha, beta);
    } else {
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
        if (d > m) m = d;
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
    double ref_err[NUM_VARIANTS] = {-1, -1, -1, -1}, max_mag = 0.0;
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
                if (d > m) m = d;
                if (fabs(ref[i]) > max_mag) max_mag = fabs(ref[i]);
            }
            ref_err[v] = m;
        }
    }

    printf("S=%d\n", S);
    for (int v = 0; v < NUM_VARIANTS; ++v) {
        printf("  %-7s %10.4f ms %9.1f GFLOPS  %6.2fx vs naive  max|x-naive| %-8.4g", VNAME[v],
               ms[v], gf[v], ms[NAIVE] / ms[v], max_abs_diff(out[v], out[NAIVE]));
        if (ref_err[v] >= 0) printf("  max|x-cpu_fp32| %.4g", ref_err[v]);
        printf("\n");
        fprintf(csv, "%d,%s,%.5f,%.2f,%.3f,%.6g,%.6g\n", S, VNAME[v], ms[v], gf[v],
                ms[NAIVE] / ms[v], max_abs_diff(out[v], out[NAIVE]),
                ref_err[v] >= 0 ? ref_err[v] : NAN);
    }
    if (S <= 1024) printf("  (max |C| = %.1f)\n", max_mag);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

int main(int argc, char** argv) {
    std::vector<int> sizes;
    for (int i = 1; i < argc; ++i) sizes.push_back(atoi(argv[i]));
    if (sizes.empty()) sizes = {256, 512, 1024, 2048, 4096};
    for (int S : sizes)
        if (S % WMMA != 0) {
            fprintf(stderr, "size %d is not a multiple of %d; the WMMA kernel needs that\n", S, WMMA);
            return 1;
        }

    if (cublasCreate(&g_handle) != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasCreate failed\n");
        return 1;
    }
    FILE* csv = fopen("../results/wmma_fp16_comparison.csv", "w");
    if (!csv) csv = fopen("wmma_fp16_comparison.csv", "w");
    fprintf(csv, "S,variant,mean_ms,gflops,speedup_vs_naive,max_abs_vs_naive,max_abs_vs_cpu_fp32\n");
    for (int S : sizes) run_size(S, csv);
    fclose(csv);
    cublasDestroy(g_handle);
    return 0;
}
