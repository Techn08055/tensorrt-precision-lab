// conv846_gemm.cu
// ResNet50 Conv_846 (the slowest layer in every TensorRT profile) as a GEMM.
// 1x1 conv, stride 2, no padding: input 1024x14x14 -> output 2048x7x7, batch 1, FP16.
//
// A 1x1 conv is a GEMM once the strided input pixels are packed:
//   out[co][p] = sum_ci W[co][ci] * X[ci][pixel p]
//   M = 2048 output channels, N = 49 output pixels, K = 1024 input channels.
// NCHW output with batch 1 is exactly the row-major M x N result, so no transpose.
//
// Its main() times the stride-2 gather, the step-4 WMMA GEMM from ../../gemm/ (at N=49 and
// with N zero-padded to 64), cuBLAS on the same packed input, and split-K with a reduce that
// adds bias (plain and with the gather folded into the tile load).
//
// This file also holds the shared conv kernel that every other driver includes (with
// CONV846_NO_MAIN defined): gemm_tile, the pipelined split-K WMMA GEMM for one output tile
// (block tile, copy width, implicit 3x3 gather and direct-output epilogue as template options),
// and its one-tile-per-block wrapper gemm_wmma_splitk_pipe. Users: conv_gemm.cu (all 53 layers),
// conv3x3_gemm.cu (the 3x3 study) and block_fused.cu (the fused bottleneck block).
//
// Build: nvcc -O3 -arch=sm_89 conv846_gemm.cu -o conv846_gemm -lcublas
// Run:   ./conv846_gemm             random data at the real shape
//        ./conv846_gemm ../data/Conv_846   real tensors from ../benchmarks/export_conv_real.py

// Reuse the step-4 kernel as-is; its benchmark main() is renamed out of the way.
#define main gemm_wmma_vec_main
#include "../../gemm/src/gemm_wmma_vec_fp16.cu"
#undef main

#include <string>
#include <cuda_pipeline.h>

#define CI 1024
#define CO 2048
#define HW_IN 14
#define HW_OUT 7
#define STRIDE 2
#define NPIX (HW_OUT * HW_OUT)
#define NPAD 64   // N padded to a multiple of 8 and of the 64-wide tile
#define NPROBE 128

// pack the stride-2 pixels into rows of width ld, zero-filling columns past 49:
// Xp[ci][oh*7+ow] = X[ci][2*oh][2*ow]
__global__ void gather_stride2(const half* X, half* Xp, int ld) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int ci = blockIdx.y;
    if (p >= ld) return;
    int oh = p / HW_OUT, ow = p % HW_OUT;
    Xp[ci * ld + p] = p < NPIX ? X[(ci * HW_IN + oh * STRIDE) * HW_IN + ow * STRIDE] : __float2half(0.0f);
}

static void launch_gather(const half* dX, half* dXp, int ld) {
    gather_stride2<<<dim3((ld + 63) / 64, CI), 64>>>(dX, dXp, ld);
}

static void launch_gemm(const half* dW, const half* dXp, half* dY, int n) {
    dim3 grid((n + BN - 1) / BN, (CO + BM - 1) / BM, 1);
    gemm_wmma_v2_kernel<true><<<grid, dim3(BLK_THREADS)>>>(dW, dXp, dY, CO, n, CI, 1.0f, 0.0f);
}

// Split-K: blockIdx.z takes one K/gridDim.z slice of the sum and writes its FP32 partial
// tile to P[z]. Same 64x64 tile, 4 warps and 2x2 fragments as the step-4 kernel.
// shortcut: assumes M%64 == 0, N%64 == 0 (pad N first) and (K/splits)%32 == 0 — true for
// Conv_846; add edge handling before reusing it on other layers.
// FOLD selects how the B tile is loaded (padding columns 49..63 always load zero):
//   0: B is a packed Kx64 buffer (from a gather/im2col kernel), 16-byte loads
//   1: B is the raw 1024x14x14 input of Conv_846, 1x1 stride-2 gather done in the load
//   2: B is the raw 512x7x7 input of Conv_850/856, 3x3 pad-1 im2col done in the load
//      (implicit GEMM, k = ci*9 + kh*3 + kw)
// AccT: accumulator type. half runs the tensor cores at 2x the FP32-accumulate rate on this
// GPU (../../gemm/profiling/mma_rate.cu) at the cost of precision; partials are stored as AccT.
template <int FOLD, typename AccT = float>
__global__ void gemm_wmma_splitk(const half* A, const half* B, AccT* P, int M, int N, int K) {
    __shared__ __align__(32) half As[BM * BK];
    __shared__ __align__(32) half Bs[BK * BN];
    int tid = threadIdx.x, warp = tid / WARP_SIZE;
    int wr = (warp / 2) * 32, wc = (warp % 2) * 32;
    int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;
    int kc = K / gridDim.z, kbeg = blockIdx.z * kc;
    // FOLD: 128 threads over a 64-wide tile, so each thread always loads the same column
    // (pixel) and its input offset can be computed once.
    int fold_p = tid % BN;
    int fold_off = fold_p < NPIX ? (fold_p / HW_OUT) * STRIDE * HW_IN + (fold_p % HW_OUT) * STRIDE : -1;
    int fold_oh = fold_p / HW_OUT, fold_ow = fold_p % HW_OUT;  // FOLD 2: 7x7 in and out

    wmma::fragment<wmma::matrix_a, WMMA, WMMA, WMMA, half, wmma::row_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, WMMA, WMMA, WMMA, half, wmma::row_major> b_frag[2];
    wmma::fragment<wmma::accumulator, WMMA, WMMA, WMMA, AccT> acc[2][2];
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++) wmma::fill_fragment(acc[i][j], (AccT)0.0f);

    for (int k0 = kbeg; k0 < kbeg + kc; k0 += BK) {
        for (int v = tid; v < BM * BK / 8; v += BLK_THREADS) {
            int r = v / (BK / 8), c = v % (BK / 8);
            ((uint4*)As)[v] = *(const uint4*)(A + (row0 + r) * K + k0 + c * 8);
        }
        if (FOLD == 1) {
            #pragma unroll
            for (int r = tid / BN; r < BK; r += BLK_THREADS / BN)
                Bs[r * BN + fold_p] = fold_off >= 0 ? B[(k0 + r) * HW_IN * HW_IN + fold_off] : __float2half(0.0f);
        } else if (FOLD == 2) {
            #pragma unroll
            for (int r = tid / BN; r < BK; r += BLK_THREADS / BN) {
                int k = k0 + r, ci = k / 9, t = k % 9;
                int ih = fold_oh + t / 3 - 1, iw = fold_ow + t % 3 - 1;
                bool in = fold_p < NPIX && ih >= 0 && ih < HW_OUT && iw >= 0 && iw < HW_OUT;
                Bs[r * BN + fold_p] = in ? B[(ci * HW_OUT + ih) * HW_OUT + iw] : __float2half(0.0f);
            }
        } else {
            for (int v = tid; v < BK * BN / 8; v += BLK_THREADS) {
                int r = v / (BN / 8), c = v % (BN / 8);
                ((uint4*)Bs)[v] = *(const uint4*)(B + (k0 + r) * N + col0 + c * 8);
            }
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < BK; kk += WMMA) {
            for (int i = 0; i < 2; i++)
                wmma::load_matrix_sync(a_frag[i], As + (wr + i * WMMA) * BK + kk, BK);
            for (int j = 0; j < 2; j++)
                wmma::load_matrix_sync(b_frag[j], Bs + kk * BN + wc + j * WMMA, BN);
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 2; j++) wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
        }
        __syncthreads();
    }
    AccT* Pz = P + (size_t)blockIdx.z * M * N;
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            wmma::store_matrix_sync(Pz + (row0 + wr + i * WMMA) * N + col0 + wc + j * WMMA, acc[i][j], N,
                                    wmma::mem_row_major);
}

// Same split-K GEMM (packed B only) with a STAGES-deep cp.async pipeline: tiles for the next
// STAGES-1 k-steps are already in flight to shared memory while the current one is computed,
// instead of load -> sync -> compute -> sync. STAGES * 8 KB of shared memory.
// PAD: extra halves per shared-memory row (A rows BK+PAD, B rows BN+PAD wide) so the 16 rows
// a fragment load reads don't all start in the same banks. Must keep rows 16-byte aligned.
// B can be read in place instead of from a 64-padded copy: ldb = B's real row stride, n_real =
// its real column count; tile columns >= n_real are zero-filled by cp.async. BVEC = halves per
// async copy (8 = 16 bytes, needs ldb % 8 == 0; 4 = 8 bytes, needs ldb % 4 == 0).
// EPI (needs gridDim.z == 1): skip the FP32 partials, add bias (+ ReLU) and write the
// final half output Y[M x n_real] straight from the accumulators.
// KS > 0: implicit GEMM. B is the raw conv input X[cin][h][h] and each B tile is gathered from the
// KS x KS windows while loading (k = ci*KS*KS + kh*KS + kw, pixel p = oh*ho + ow), so no im2col
// buffer. The gather is plain loads (window offsets break 16-byte alignment); A stays on cp.async.
// To hide those loads, tile t+2 is gathered into registers while tile t is computed, and written to
// shared memory right after (register prefetch), so B runs one tile ahead instead of stalling.
// Shared memory one tile of gemm_tile needs: STAGES A + B tiles, or the epilogue scratch, whichever is bigger.
template <int STAGES, int PAD, bool EPI, int TM, int TN>
__host__ __device__ constexpr int pipe_smem_bytes() {
    constexpr int a = STAGES * TM * (BK + PAD) * 2, b = STAGES * BK * (TN + PAD) * 2;
    constexpr int c = EPI ? WARPS_PER_BLOCK * (TM / 2) * (TN / 2) * 4 : 0;
    return a + b > c ? a + b : c;
}

// One output tile (bx, by) and K-slice bz of gz: the whole kernel body as a device function, so a
// persistent kernel can run many tiles per block (fused bottleneck block) and the per-layer kernel
// below is a thin wrapper. smem: pipe_smem_bytes<...>() bytes, 32-byte aligned.
// resid (EPI only): optional half residual [M x n_real] added before the ReLU.
template <int STAGES, typename AccT = float, int PAD = 0, int BVEC = 8, bool EPI = false, int KS = 0, int TM = BM,
          int TN = BN>
__device__ __forceinline__ void gemm_tile(unsigned char* smem, int bx, int by, int bz, int gz, const half* A,
                                          const half* B, AccT* P, int M, int N, int K, int ldb, int n_real, half* Y,
                                          const float* bias, bool relu, int cin, int h, int cs, int cpad, int ho, int kr,
                                          const half* resid = nullptr) {
    // TM x TN block tile (64 or 32 each): 4 warps in 2x2, each warp a (TM/2) x (TN/2) patch of FI x FJ fragments
    constexpr int WM = TM / 2, WN = TN / 2, FI = WM / WMMA, FJ = WN / WMMA;
    constexpr int LDA = BK + PAD, LDB = TN + PAD;
    constexpr int A_BYTES = STAGES * TM * LDA * 2;
    __syncthreads();  // the previous tile run by this block may still be reading smem
    auto As = reinterpret_cast<half(*)[TM * LDA]>(smem);
    auto Bs = reinterpret_cast<half(*)[BK * LDB]>(smem + A_BYTES);
    if (ldb < 0) ldb = N;
    if (n_real < 0) n_real = N;
    int tid = threadIdx.x, warp = tid / WARP_SIZE;
    int wr = (warp / 2) * WM, wc = (warp % 2) * WN;
    int row0 = by * TM, col0 = bx * TN;
    int kc = K / gz, kbeg = bz * kc, ntiles = kc / BK;
    // implicit GEMM: 128 threads over a TN-wide tile, so each thread always gathers the same pixel column
    const int gp = col0 + tid % TN;
    const int gih = (gp / ho) * cs - cpad, giw = (gp % ho) * cs - cpad;
    const bool gvalid = gp < n_real;

    // Each thread's async copies are the same every k-tile apart from a fixed step, so their source
    // pointers and shared-memory offsets are worked out once here; issue() only adds t * step.
    // (Nsight Compute: recomputing them per copy was most of our 1.58x extra instructions vs TensorRT.)
    constexpr int NA = TM * BK / 8 / BLK_THREADS;                          // 16-byte A copies per thread
    constexpr int NB = KS == 0 ? BK * TN / BVEC / BLK_THREADS : 1;         // B copies per thread
    const half* a_src[NA];
    int a_dst[NA];
    #pragma unroll
    for (int i = 0; i < NA; ++i) {
        int v = tid + i * BLK_THREADS, r = v / (BK / 8), c = v % (BK / 8);
        a_src[i] = A + (size_t)(row0 + r) * K + kbeg + c * 8;
        a_dst[i] = r * LDA + c * 8;
    }
    const half* b_src[NB];
    int b_dst[NB], b_zfill[NB];
    if constexpr (KS == 0) {
        #pragma unroll
        for (int i = 0; i < NB; ++i) {
            int v = tid + i * BLK_THREADS, r = v / (TN / BVEC), c = v % (TN / BVEC), gc = col0 + c * BVEC;
            bool ok = gc < n_real;  // columns past n_real: zero-fill the whole copy, read nothing
            b_src[i] = B + (size_t)(kbeg + r) * ldb + (ok ? gc : 0);
            b_dst[i] = r * LDB + c * BVEC;
            b_zfill[i] = ok ? 0 : BVEC * 2;
        }
    }
    const size_t b_step = (size_t)BK * ldb;

    auto issue = [&](int t) {  // async-copy k-tile t into stage t % STAGES
        if (t < ntiles) {
            int s = t % STAGES;
            #pragma unroll
            for (int i = 0; i < NA; ++i) __pipeline_memcpy_async(&As[s][a_dst[i]], a_src[i] + t * BK, 16);
            if constexpr (KS == 0) {  // implicit B is gathered by gather()/stash() below
                #pragma unroll
                for (int i = 0; i < NB; ++i)
                    __pipeline_memcpy_async(&Bs[s][b_dst[i]], b_src[i] + t * b_step, BVEC * 2, b_zfill[i]);
            }
        }
        __pipeline_commit();  // commit even when empty so the wait count below stays uniform
    };

    // two sets of fragments: the next k-slice's ldmatrix loads go out before this slice's mma_syncs
    wmma::fragment<wmma::matrix_a, WMMA, WMMA, WMMA, half, wmma::row_major> a_frag[2][FI];
    wmma::fragment<wmma::matrix_b, WMMA, WMMA, WMMA, half, wmma::row_major> b_frag[2][FJ];
    wmma::fragment<wmma::accumulator, WMMA, WMMA, WMMA, AccT> acc[FI][FJ];
    for (int i = 0; i < FI; i++)
        for (int j = 0; j < FJ; j++) wmma::fill_fragment(acc[i][j], (AccT)0.0f);

    constexpr int GR = KS > 0 ? BK * TN / BLK_THREADS : 1;  // B values each thread gathers per tile
    constexpr int KK = KS > 0 ? KS : 1;                     // keeps the unused KS = 0 build free of /0
    half breg[GR];
    auto gather = [&](int t) {  // implicit GEMM: load tile t's B values into registers
        if (t >= ntiles) return;
        #pragma unroll
        for (int i = 0; i < GR; ++i) {
            int kq = kbeg + t * BK + tid / TN + i * (BLK_THREADS / TN), ci = kq / (KK * KK), tap = kq % (KK * KK);
            int ih = gih + tap / KK, iw = giw + tap % KK;
            bool in = gvalid && kq < kr && ih >= 0 && ih < h && iw >= 0 && iw < h;
            breg[i] = in ? B[((size_t)ci * h + ih) * h + iw] : __float2half(0.0f);
        }
    };
    auto stash = [&](int t) {  // ...and write them into tile t's shared-memory stage
        if (t >= ntiles) return;
        #pragma unroll
        for (int i = 0; i < GR; ++i) Bs[t % STAGES][(tid / TN + i * (BLK_THREADS / TN)) * LDB + tid % TN] = breg[i];
    };
    if constexpr (KS > 0) { gather(0); stash(0); gather(1); }

    for (int t = 0; t < STAGES - 1; ++t) issue(t);
    for (int t = 0; t < ntiles; ++t) {
        __pipeline_wait_prior(STAGES - 2);  // tile t has landed (this thread's copies)
        __syncthreads();                    // ...and everyone's; also: all warps are done with tile t-1
        issue(t + STAGES - 1);              // refill the stage tile t-1 used
        const half* as = As[t % STAGES];
        const half* bs = Bs[t % STAGES];
        auto load_frags = [&](int buf, int kk) {
            #pragma unroll
            for (int i = 0; i < FI; i++) wmma::load_matrix_sync(a_frag[buf][i], as + (wr + i * WMMA) * LDA + kk, LDA);
            #pragma unroll
            for (int j = 0; j < FJ; j++) wmma::load_matrix_sync(b_frag[buf][j], bs + kk * LDB + wc + j * WMMA, LDB);
        };
        load_frags(0, 0);
        #pragma unroll
        for (int q = 0; q < BK / WMMA; ++q) {
            if (q + 1 < BK / WMMA) load_frags((q + 1) & 1, (q + 1) * WMMA);
            #pragma unroll
            for (int i = 0; i < FI; i++)
                #pragma unroll
                for (int j = 0; j < FJ; j++) wmma::mma_sync(acc[i][j], a_frag[q & 1][i], b_frag[q & 1][j], acc[i][j]);
        }
        // stage (t+1) % STAGES was last read for tile t+1-STAGES, which every warp finished before the
        // barrier above, so it's free; the next barrier makes these stores visible before tile t+1 is read
        if constexpr (KS > 0) { stash(t + 1); gather(t + 2); }
    }
    if constexpr (EPI) {
        __pipeline_wait_prior(0);
        __syncthreads();  // every warp is done reading the tiles, so their memory can hold the scratch
        AccT* Cw = reinterpret_cast<AccT*>(smem) + warp * WM * WN;  // this warp's WM x WN outputs
        for (int i = 0; i < FI; i++)
            for (int j = 0; j < FJ; j++)
                wmma::store_matrix_sync(Cw + i * WMMA * WN + j * WMMA, acc[i][j], WN, wmma::mem_row_major);
        __syncwarp();
        int lane = tid % WARP_SIZE;
        for (int e = lane; e < WM * WN; e += WARP_SIZE) {  // consecutive lanes -> consecutive pixels
            int co = row0 + wr + e / WN, p = col0 + wc + e % WN;
            if (p < n_real) {
                float v = (float)Cw[e] + bias[co];
                if (resid) v += __half2float(resid[(size_t)co * n_real + p]);
                Y[(size_t)co * n_real + p] = __float2half(relu ? fmaxf(v, 0.0f) : v);
            }
        }
        return;
    }
    AccT* Pz = P + (size_t)bz * M * N;
    for (int i = 0; i < FI; i++)
        for (int j = 0; j < FJ; j++)
            wmma::store_matrix_sync(Pz + (row0 + wr + i * WMMA) * N + col0 + wc + j * WMMA, acc[i][j], N,
                                    wmma::mem_row_major);
}

template <int STAGES, typename AccT = float, int PAD = 0, int BVEC = 8, bool EPI = false, int KS = 0, int TM = BM,
          int TN = BN>
__global__ void gemm_wmma_splitk_pipe(const half* A, const half* B, AccT* P, int M, int N, int K, int ldb = -1,
                                      int n_real = -1, half* Y = nullptr, const float* bias = nullptr, bool relu = false,
                                      int cin = 0, int h = 0, int cs = 1, int cpad = 0, int ho = 1, int kr = 0) {
    __shared__ __align__(32) unsigned char smem[pipe_smem_bytes<STAGES, PAD, EPI, TM, TN>()];
    gemm_tile<STAGES, AccT, PAD, BVEC, EPI, KS, TM, TN>(smem, blockIdx.x, blockIdx.y, blockIdx.z, gridDim.z, A, B, P, M,
                                                        N, K, ldb, n_real, Y, bias, relu, cin, h, cs, cpad, ho, kr);
}

// Sum the split-K partials, add bias, drop the padding columns: Y is plain NCHW 2048x7x7.
__global__ void splitk_reduce_bias(const float* P, const float* bias, half* Y, int splits) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= CO * NPIX) return;
    int co = i / NPIX, p = i % NPIX;
    float s = bias[co];
    for (int z = 0; z < splits; ++z) s += P[((size_t)z * CO + co) * NPAD + p];
    Y[i] = __float2half(s);
}

template <int FOLD>
static void launch_splitk(const half* dW, const half* dXp, float* dP, const float* dBias, half* dY, int splits) {
    gemm_wmma_splitk<FOLD><<<dim3(NPAD / BN, CO / BM, splits), dim3(BLK_THREADS)>>>(dW, dXp, dP, CO, NPAD, CI);
    splitk_reduce_bias<<<(CO * NPIX + 255) / 256, 256>>>(dP, dBias, dY, splits);
}

static void launch_cublas(const half* dW, const half* dXp, half* dY) {
    const float alpha = 1.0f, beta = 0.0f;
    // row-major Y = W * Xp is column-major Y^T = Xp^T * W^T (see the gemm/ driver)
    cublasStatus_t st = cublasGemmEx(g_handle, CUBLAS_OP_N, CUBLAS_OP_N, NPIX, CO, CI, &alpha,
                                     dXp, CUDA_R_16F, NPIX, dW, CUDA_R_16F, CI, &beta, dY,
                                     CUDA_R_16F, NPIX, CUBLAS_COMPUTE_32F,
                                     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (st != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cublasGemmEx failed: %d\n", (int)st); exit(1); }
}

template <typename F>
static float time_ms(F f, int warmup = 5, int iters = 50) {
    for (int i = 0; i < warmup; ++i) f();
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t s, e;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&e));
    CUDA_CHECK(cudaEventRecord(s));
    for (int i = 0; i < iters; ++i) f();
    CUDA_CHECK(cudaEventRecord(e));
    CUDA_CHECK(cudaEventSynchronize(e));
    CUDA_CHECK(cudaGetLastError());
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
    cudaEventDestroy(s);
    cudaEventDestroy(e);
    return ms / iters;  // back-to-back launches, so this includes launch gaps
}

template <typename T>
static void read_bin(const std::string& path, std::vector<T>& v) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f || fread(v.data(), sizeof(T), v.size(), f) != v.size()) {
        fprintf(stderr, "cannot read %zu values from %s (run export_conv_real.py)\n", v.size(), path.c_str());
        exit(1);
    }
    fclose(f);
}

#ifndef CONV846_NO_MAIN  // conv_gemm.cu, conv3x3_gemm.cu and block_fused.cu include this file for its kernels
int main(int argc, char** argv) {
    const bool real = argc > 1;
    std::vector<half> hX((size_t)CI * HW_IN * HW_IN), hW((size_t)CO * CI), hY((size_t)CO * NPAD);
    std::vector<float> hB(CO, 0.0f), hYref;
    if (real) {
        std::string dir = argv[1];
        hYref.resize((size_t)CO * NPIX);
        read_bin(dir + "/x.bin", hX);
        read_bin(dir + "/w.bin", hW);
        read_bin(dir + "/b.bin", hB);
        read_bin(dir + "/y_ref.bin", hYref);
    } else {
        srand(846);
        for (auto& v : hX) v = __float2half((float)rand() / RAND_MAX * 2.0f - 1.0f);
        for (auto& v : hW) v = __float2half(((float)rand() / RAND_MAX * 2.0f - 1.0f) * 0.05f);
    }

    half *dX, *dXp, *dW, *dY;
    CUDA_CHECK(cudaMalloc(&dX, hX.size() * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dXp, (size_t)CI * NPROBE * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dW, hW.size() * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dY, (size_t)CO * NPROBE * sizeof(half)));
    CUDA_CHECK(cudaMemcpy(dX, hX.data(), hX.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dW, hW.data(), hW.size() * sizeof(half), cudaMemcpyHostToDevice));
    if (cublasCreate(&g_handle) != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cublasCreate failed\n"); return 1; }

    // CPU fp32 conv from the same fp16 inputs, straight from the conv definition
    std::vector<float> ref((size_t)CO * NPIX, 0.0f);
    for (int co = 0; co < CO; ++co)
        for (int ci = 0; ci < CI; ++ci) {
            float w = __half2float(hW[(size_t)co * CI + ci]);
            for (int p = 0; p < NPIX; ++p) {
                int oh = p / HW_OUT, ow = p % HW_OUT;
                ref[(size_t)co * NPIX + p] +=
                    w * __half2float(hX[((size_t)ci * HW_IN + oh * STRIDE) * HW_IN + ow * STRIDE]);
            }
        }

    // compare dY (row stride ld) + bias against the CPU conv and, with real data, ONNX's output
    auto check = [&](const char* name, int ld) {
        CUDA_CHECK(cudaMemcpy(hY.data(), dY, (size_t)CO * ld * sizeof(half), cudaMemcpyDeviceToHost));
        double e_cpu = 0.0, e_onnx = 0.0, mag = 0.0;
        for (int co = 0; co < CO; ++co)
            for (int p = 0; p < NPIX; ++p) {
                double out = __half2float(hY[(size_t)co * ld + p]);
                double d = fabs(out - ref[(size_t)co * NPIX + p]);
                if (!(d <= e_cpu)) e_cpu = d;  // also catches NaN
                if (real) {
                    double r = hYref[(size_t)co * NPIX + p];
                    double d2 = fabs(out + hB[co] - r);
                    if (!(d2 <= e_onnx)) e_onnx = d2;
                    mag = fmax(mag, fabs(r));
                }
            }
        printf("  %-12s max|out - cpu_fp32| %.4g", name, e_cpu);
        if (real) printf("   max|out + bias - onnx_fp32| %.4g  (max|onnx| %.2f)", e_onnx, mag);
        printf("\n");
        return real ? e_onnx : e_cpu;
    };

    printf("Conv_846 (%s data): 1x1 stride-2 conv, %dx%dx%d -> %dx%dx%d, as GEMM M=%d N=%d K=%d\n",
           real ? "real ResNet50" : "random", CI, HW_IN, HW_IN, CO, HW_OUT, HW_OUT, CO, NPIX, CI);
    printf("correctness:\n");
    launch_gather(dX, dXp, NPIX);
    launch_gemm(dW, dXp, dY, NPIX);
    double err_wmma = check("wmma N=49", NPIX);
    launch_cublas(dW, dXp, dY);
    double err_cublas = check("cublas", NPIX);
    launch_gather(dX, dXp, NPAD);
    launch_gemm(dW, dXp, dY, NPAD);
    double err_pad = check("wmma N=64pad", NPAD);

    // time each piece; the padded buffer is rebuilt by its own gather inside its timing loop
    float t_gather = time_ms([&] { launch_gather(dX, dXp, NPIX); });
    float t_wmma = time_ms([&] { launch_gemm(dW, dXp, dY, NPIX); });
    float t_cublas = time_ms([&] { launch_cublas(dW, dXp, dY); });
    float t_wmma_all = time_ms([&] { launch_gather(dX, dXp, NPIX); launch_gemm(dW, dXp, dY, NPIX); });
    float t_cublas_all = time_ms([&] { launch_gather(dX, dXp, NPIX); launch_cublas(dW, dXp, dY); });
    float t_pad_all = time_ms([&] { launch_gather(dX, dXp, NPAD); launch_gemm(dW, dXp, dY, NPAD); });
    // probe: 2x the columns -> 2x the blocks and 2x the work. Same time => the GPU was underfilled.
    float t_pad = time_ms([&] { launch_gemm(dW, dXp, dY, NPAD); });
    float t_probe = time_ms([&] { launch_gemm(dW, dXp, dY, NPROBE); });

    // split-K sweep: gather (padded) + split-K GEMM + reduce with bias, output is final NCHW
    const int split_opts[] = {1, 2, 4, 8, 16, 32};
    const int NSPLIT = sizeof(split_opts) / sizeof(int);
    float* dP;
    float* dBias;
    CUDA_CHECK(cudaMalloc(&dP, (size_t)32 * CO * NPAD * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dBias, CO * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dBias, hB.data(), CO * sizeof(float), cudaMemcpyHostToDevice));
    // [0] = gather kernel + split-K on the packed buffer, [1] = gather folded into split-K (reads dX)
    float t_split[2][NSPLIT];
    double e_split[2][NSPLIT];
    auto run_variant = [&](int fold, int splits) {
        if (fold) launch_splitk<1>(dW, dX, dP, dBias, dY, splits);
        else { launch_gather(dX, dXp, NPAD); launch_splitk<0>(dW, dXp, dP, dBias, dY, splits); }
    };
    for (int f = 0; f < 2; ++f)
    for (int s = 0; s < NSPLIT; ++s) {
        CUDA_CHECK(cudaMemset(dY, 0, (size_t)CO * NPIX * sizeof(half)));
        run_variant(f, split_opts[s]);
        CUDA_CHECK(cudaMemcpy(hY.data(), dY, (size_t)CO * NPIX * sizeof(half), cudaMemcpyDeviceToHost));
        // bias is already in the output here
        e_split[f][s] = 0.0;
        for (size_t i = 0; i < ref.size(); ++i) {
            double r = real ? hYref[i] : ref[i] + hB[i / NPIX];
            double d = fabs((double)__half2float(hY[i]) - r);
            if (!(d <= e_split[f][s])) e_split[f][s] = d;
        }
        t_split[f][s] = time_ms([&] { run_variant(f, split_opts[s]); });
    }

    const double flops = 2.0 * CO * NPIX * CI;  // useful work only, also for the padded run
    const double trt_fp16_ms = 0.043;  // TensorRT FP16 layer time, ../../profiling/PROFILE_RESULTS.md
    printf("timing (mean of 50 back-to-back launches; GFLOPS count the 49 real columns only):\n");
    printf("  gather only             %8.4f ms\n", t_gather);
    printf("  wmma GEMM N=49          %8.4f ms  %8.1f GFLOPS\n", t_wmma, flops / t_wmma / 1e6);
    printf("  cublas GEMM N=49        %8.4f ms  %8.1f GFLOPS\n", t_cublas, flops / t_cublas / 1e6);
    printf("  gather + wmma N=49      %8.4f ms  %8.1f GFLOPS\n", t_wmma_all, flops / t_wmma_all / 1e6);
    printf("  gather + wmma N=64pad   %8.4f ms  %8.1f GFLOPS\n", t_pad_all, flops / t_pad_all / 1e6);
    printf("  gather + cublas         %8.4f ms  %8.1f GFLOPS\n", t_cublas_all, flops / t_cublas_all / 1e6);
    printf("  TensorRT FP16 layer     %8.4f ms  %8.1f GFLOPS  (from the per-layer profile, includes bias)\n",
           trt_fp16_ms, flops / trt_fp16_ms / 1e6);
    printf("probe: wmma GEMM N=64 (%d blocks) %.4f ms vs N=128 (%d blocks) %.4f ms -> %.2fx time for 2x work\n",
           CO / BM, t_pad, 2 * CO / BM, t_probe, t_probe / t_pad);
    const char* vname[2] = {"gather + split-K + reduce/bias", "folded split-K + reduce/bias"};
    for (int f = 0; f < 2; ++f) {
        printf("%s (final NCHW output):\n", vname[f]);
        for (int s = 0; s < NSPLIT; ++s)
            printf("  splits=%-2d (%4d blocks)  %8.4f ms  %8.1f GFLOPS  %5.2fx vs cublas path  max|out - %s| %.4g\n",
                   split_opts[s], split_opts[s] * CO / BM, t_split[f][s], flops / t_split[f][s] / 1e6,
                   t_cublas_all / t_split[f][s], real ? "onnx_fp32" : "cpu_fp32+bias", e_split[f][s]);
    }

    FILE* csv = fopen(real ? "../results/conv846_gemm_real.csv" : "../results/conv846_gemm.csv", "w");
    if (!csv) csv = fopen("conv846_gemm.csv", "w");
    fprintf(csv, "variant,mean_ms,gflops,max_abs_err\n");
    fprintf(csv, "gather,%.5f,,\n", t_gather);
    fprintf(csv, "wmma_gemm_n49,%.5f,%.1f,%.6g\n", t_wmma, flops / t_wmma / 1e6, err_wmma);
    fprintf(csv, "cublas_gemm,%.5f,%.1f,%.6g\n", t_cublas, flops / t_cublas / 1e6, err_cublas);
    fprintf(csv, "gather_plus_wmma_n49,%.5f,%.1f,%.6g\n", t_wmma_all, flops / t_wmma_all / 1e6, err_wmma);
    fprintf(csv, "gather_plus_wmma_n64pad,%.5f,%.1f,%.6g\n", t_pad_all, flops / t_pad_all / 1e6, err_pad);
    fprintf(csv, "gather_plus_cublas,%.5f,%.1f,%.6g\n", t_cublas_all, flops / t_cublas_all / 1e6, err_cublas);
    fprintf(csv, "tensorrt_fp16_profile,%.5f,%.1f,\n", trt_fp16_ms, flops / trt_fp16_ms / 1e6);
    fprintf(csv, "probe_wmma_n64,%.5f,,\n", t_pad);
    fprintf(csv, "probe_wmma_n128,%.5f,,\n", t_probe);
    for (int f = 0; f < 2; ++f)
        for (int s = 0; s < NSPLIT; ++s)
            fprintf(csv, "%s%d_bias,%.5f,%.1f,%.6g\n", f ? "folded_splitk" : "gather_plus_splitk", split_opts[s],
                    t_split[f][s], flops / t_split[f][s] / 1e6, e_split[f][s]);
    fclose(csv);

    cublasDestroy(g_handle);
    return 0;
}
#endif
