// gemm_regblock_fp16.cu
// Register-blocked tiled GEMM, FP16 in/out, FP32 accumulation, C = alpha*A*B + beta*C.
// Same 32x32 shared-memory tile as gemm_tiled_fp16.cu, but each thread computes an
// RxR patch of C instead of one element, so the thread block shrinks to
// (32/R)x(32/R). The tiled kernel is copied here unchanged as the baseline, so
// register blocking and the smaller thread block are the only differences.
//
// Build: nvcc -O3 -arch=sm_89 gemm_regblock_fp16.cu -o gemm_regblock_fp16
// Run:   ./gemm_regblock_fp16 [sizes...]     (default: 256 512 1000 1024 2048 4096)

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

#define TILE_SIZE 32

// --- baseline: the submitted tiled kernel, unchanged --------------------------
__global__ void matrix_multiplication_kernel_simp( const half * A, const half * B, half * C, int M, int N, int K, float alpha, float beta){
    __shared__ float Ashared[TILE_SIZE][TILE_SIZE];
    __shared__ float Bshared[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    float Pvalue1 = 0;
    int iters = (K + TILE_SIZE -1)/TILE_SIZE;
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

// --- register-blocked kernel --------------------------------------------------
// R = patch size per thread (R x R outputs). Launch with a (32/R) x (32/R) block.
template <int R>
__global__ void gemm_regblock_kernel(const half* A, const half* B, half* C, int M, int N, int K,
                                     float alpha, float beta) {
    __shared__ float Ashared[TILE_SIZE][TILE_SIZE];
    __shared__ float Bshared[TILE_SIZE][TILE_SIZE];

    // top-left corner of this thread's patch, inside the tile and in the full matrix
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
        // each thread now fills an RxR piece of both tiles (R*R times fewer threads)
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
            // 2R shared-memory reads feed R*R multiply-adds
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

    // same epilogue as the tiled kernel (including the read of C when beta == 0)
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

#include <cstdlib>
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

// Naive baseline with the same contract: one thread per C element, FP16 loads
// straight from global memory, FP32 accumulate, same epilogue.
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

enum Variant { NAIVE, TILED, RB2, RB4, RB8, NUM_VARIANTS };
static const char* kNames[NUM_VARIANTS] = {"naive", "tiled", "rb2x2", "rb4x4", "rb8x8"};

static void launch(Variant v, const half* dA, const half* dB, half* dC, int M, int N, int K) {
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE, 1);
    switch (v) {
        case NAIVE: gemm_naive_fp16<<<grid, dim3(32, 32)>>>(dA, dB, dC, M, N, K, 1.0f, 0.0f); break;
        case TILED: matrix_multiplication_kernel_simp<<<grid, dim3(32, 32)>>>(dA, dB, dC, M, N, K, 1.0f, 0.0f); break;
        case RB2:   gemm_regblock_kernel<2><<<grid, dim3(16, 16)>>>(dA, dB, dC, M, N, K, 1.0f, 0.0f); break;
        case RB4:   gemm_regblock_kernel<4><<<grid, dim3(8, 8)>>>(dA, dB, dC, M, N, K, 1.0f, 0.0f); break;
        case RB8:   gemm_regblock_kernel<8><<<grid, dim3(4, 4)>>>(dA, dB, dC, M, N, K, 1.0f, 0.0f); break;
        default: break;
    }
}

static float time_kernel(Variant v, const half* dA, const half* dB, half* dC, int M, int N, int K,
                         int warmup, int iters) {
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

static void run_size(int S, FILE* csv) {
    const int M = S, N = S, K = S;
    std::vector<half> hA((size_t)M * K), hB((size_t)K * N);
    std::vector<half> hC[NUM_VARIANTS];
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

    const int warmup = 2, iters = 5;
    float ms[NUM_VARIANTS];
    for (int v = 0; v < NUM_VARIANTS; ++v) {
        hC[v].resize((size_t)M * N);
        CUDA_CHECK(cudaMemset(dC, 0, bC));
        ms[v] = time_kernel((Variant)v, dA, dB, dC, M, N, K, warmup, iters);
        CUDA_CHECK(cudaMemcpy(hC[v].data(), dC, bC, cudaMemcpyDeviceToHost));
    }

    // every variant vs naive
    double max_vs_naive[NUM_VARIANTS] = {0};
    for (int v = 1; v < NUM_VARIANTS; ++v)
        for (size_t i = 0; i < hC[v].size(); ++i) {
            double d = fabs((double)__half2float(hC[v][i]) - (double)__half2float(hC[NAIVE][i]));
            if (!(d <= max_vs_naive[v])) max_vs_naive[v] = d;  // also catches NaN
        }

    // CPU fp32 reference (same half inputs) for sizes small enough to finish quickly
    double max_ref[NUM_VARIANTS];
    for (int v = 0; v < NUM_VARIANTS; ++v) max_ref[v] = -1.0;
    if (S <= 1024) {
        std::vector<float> fA(hA.size()), fB(hB.size());
        for (size_t i = 0; i < hA.size(); ++i) fA[i] = __half2float(hA[i]);
        for (size_t i = 0; i < hB.size(); ++i) fB[i] = __half2float(hB[i]);
        std::vector<float> ref((size_t)M * N, 0.0f);
        for (int i = 0; i < M; ++i)
            for (int k = 0; k < K; ++k) {
                float a = fA[(size_t)i * K + k];
                const float* brow = &fB[(size_t)k * N];
                float* crow = &ref[(size_t)i * N];
                for (int j = 0; j < N; ++j) crow[j] += a * brow[j];
            }
        for (int v = 0; v < NUM_VARIANTS; ++v) {
            max_ref[v] = 0.0;
            for (size_t i = 0; i < ref.size(); ++i) {
                double d = fabs((double)__half2float(hC[v][i]) - (double)ref[i]);
                if (!(d <= max_ref[v])) max_ref[v] = d;
            }
        }
    }

    double flops = 2.0 * M * N * K;
    printf("S=%d\n", S);
    for (int v = 0; v < NUM_VARIANTS; ++v) {
        double gf = flops / (ms[v] / 1e3) / 1e9;
        printf("  %-6s %9.4f ms %8.1f GFLOPS  x%5.2f vs naive  x%5.2f vs tiled  max|d naive| %.4g",
               kNames[v], ms[v], gf, ms[NAIVE] / ms[v], ms[TILED] / ms[v], max_vs_naive[v]);
        if (max_ref[v] >= 0) printf("  max|d cpu_fp32| %.4g", max_ref[v]);
        printf("\n");
        fprintf(csv, "%d,%s,%.5f,%.2f,%.3f,%.3f,%.6g,%.6g\n", S, kNames[v], ms[v], gf,
                ms[NAIVE] / ms[v], ms[TILED] / ms[v], max_vs_naive[v],
                max_ref[v] >= 0 ? max_ref[v] : NAN);
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

int main(int argc, char** argv) {
    std::vector<int> sizes;
    for (int i = 1; i < argc; ++i) sizes.push_back(atoi(argv[i]));
    if (sizes.empty()) sizes = {256, 512, 1000, 1024, 2048, 4096};

    FILE* csv = fopen("../results/regblock_fp16_comparison.csv", "w");
    if (!csv) csv = fopen("regblock_fp16_comparison.csv", "w");
    fprintf(csv, "S,kernel,mean_ms,gflops,speedup_vs_naive,speedup_vs_tiled,max_abs_vs_naive,"
                 "max_abs_vs_cpu_fp32\n");
    for (int S : sizes) run_size(S, csv);
    fclose(csv);
    return 0;
}
