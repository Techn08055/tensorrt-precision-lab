// block_fused.cu
// One whole ResNet50 bottleneck block (last stage, 7x7), batch 1, FP16, as a single persistent kernel:
//   x[2048x7x7] -> Conv_848 1x1 -> ReLU -> Conv_850 3x3 -> ReLU -> Conv_852 1x1 -> + x -> ReLU -> out
//
// Stages (grid-wide sync between each, one cooperative launch):
//   0  copy x into the 64-column padded layout Xp[2048][64]
//   1  GEMM  W1[512x2048] x Xp            split-K S1 -> FP32 partials P1
//   2  reduce P1 + b1, ReLU, and scatter each value straight into the im2col layout the 3x3 needs
//      (Xc[4608][64]), so there is no separate im2col pass
//   3  GEMM  W2[512x4608] x Xc            split-K S2 -> FP32 partials P2
//   4  reduce P2 + b2, ReLU -> t2[512][64]
//   5  GEMM  W3[2048x512] x t2, 1 split, epilogue adds b3 + residual x, ReLU -> out[2048][49] (NCHW)
// The same stage code also runs as one kernel launch per stage ("separate"), to split what the single
// launch buys from what the layout changes buy.
//
// Build: nvcc -O3 -arch=sm_89 block_fused.cu -o block_fused -lcublas
// Run:   ./block_fused [S1 S2]     (data from ../benchmarks/export_conv_real.py and trt_block.py)

#define CONV846_NO_MAIN
#include "conv846_gemm.cu"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define C0 2048  // block in/out channels
#define C1 512   // bottleneck channels
#define HW 7
#define NP 49    // pixels
#define NC 64    // padded columns

struct Block {
    const half *x, *w1, *w2, *w3;
    const float *b1, *b2, *b3;
    half *xp, *xc, *t2, *out;
    float *p1, *p2;
    int s1, s2;
};

constexpr int SMEM = pipe_smem_bytes<2, 8, false, 64, 64>() > pipe_smem_bytes<2, 8, true, 64, 64>()
                         ? pipe_smem_bytes<2, 8, false, 64, 64>()
                         : pipe_smem_bytes<2, 8, true, 64, 64>();

// each stage: virtual block v of nv, so it runs the same way in one persistent grid or its own launch
__device__ void stage0(const Block& b, int v, int nv) {
    for (int i = v * BLK_THREADS + threadIdx.x; i < C0 * NP; i += nv * BLK_THREADS)
        b.xp[(i / NP) * NC + i % NP] = b.x[i];  // columns 49..63 stay zero from the one-time memset
}
__device__ void stage1(const Block& b, unsigned char* smem, int v, int nv) {
    for (int t = v; t < (C1 / 64) * b.s1; t += nv)
        gemm_tile<2, float, 8, 8, false, 0, 64, 64>(smem, 0, t % (C1 / 64), t / (C1 / 64), b.s1, b.w1, b.xp, b.p1, C1,
                                                    NC, C0, NC, NP, nullptr, nullptr, false, 0, 0, 1, 0, 1, 0);
}
__device__ void stage2(const Block& b, int v, int nv) {
    for (int i = v * BLK_THREADS + threadIdx.x; i < C1 * NP; i += nv * BLK_THREADS) {
        int c = i / NP, p = i % NP;
        float s = b.b1[c];
        for (int z = 0; z < b.s1; ++z) s += b.p1[((size_t)z * C1 + c) * NC + p];
        half v1 = __float2half(fmaxf(s, 0.0f));
        // t1[c][ih][iw] feeds output pixel (oh, ow) = (ih - kh + 1, iw - kw + 1) through tap (kh, kw)
        int ih = p / HW, iw = p % HW;
        for (int tap = 0; tap < 9; ++tap) {
            int oh = ih - tap / 3 + 1, ow = iw - tap % 3 + 1;
            if (oh >= 0 && oh < HW && ow >= 0 && ow < HW) b.xc[(size_t)(c * 9 + tap) * NC + oh * HW + ow] = v1;
        }  // taps that fall in the zero-padding ring are never written and stay zero
    }
}
__device__ void stage3(const Block& b, unsigned char* smem, int v, int nv) {
    for (int t = v; t < (C1 / 64) * b.s2; t += nv)
        gemm_tile<2, float, 8, 8, false, 0, 64, 64>(smem, 0, t % (C1 / 64), t / (C1 / 64), b.s2, b.w2, b.xc, b.p2, C1,
                                                    NC, C1 * 9, NC, NP, nullptr, nullptr, false, 0, 0, 1, 0, 1, 0);
}
__device__ void stage4(const Block& b, int v, int nv) {
    for (int i = v * BLK_THREADS + threadIdx.x; i < C1 * NP; i += nv * BLK_THREADS) {
        int c = i / NP, p = i % NP;
        float s = b.b2[c];
        for (int z = 0; z < b.s2; ++z) s += b.p2[((size_t)z * C1 + c) * NC + p];
        b.t2[c * NC + p] = __float2half(fmaxf(s, 0.0f));
    }
}
__device__ void stage5(const Block& b, unsigned char* smem, int v, int nv) {
    for (int t = v; t < C0 / 64; t += nv)
        gemm_tile<2, float, 8, 8, true, 0, 64, 64>(smem, 0, t, 0, 1, b.w3, b.t2, nullptr, C0, NC, C1, NC, NP, b.out,
                                                   b.b3, true, 0, 0, 1, 0, 1, 0, b.x);
}

__global__ void __launch_bounds__(BLK_THREADS) block_fused(Block b) {
    __shared__ __align__(32) unsigned char smem[SMEM];
    cg::grid_group grid = cg::this_grid();
    int v = blockIdx.x, nv = gridDim.x;
    stage0(b, v, nv); grid.sync();
    stage1(b, smem, v, nv); grid.sync();
    stage2(b, v, nv); grid.sync();
    stage3(b, smem, v, nv); grid.sync();
    stage4(b, v, nv); grid.sync();
    stage5(b, smem, v, nv);
}

// the same stages, one launch each
__global__ void k_stage(Block b, int which) {
    __shared__ __align__(32) unsigned char smem[SMEM];
    int v = blockIdx.x, nv = gridDim.x;
    switch (which) {
        case 0: stage0(b, v, nv); break;
        case 1: stage1(b, smem, v, nv); break;
        case 2: stage2(b, v, nv); break;
        case 3: stage3(b, smem, v, nv); break;
        case 4: stage4(b, v, nv); break;
        default: stage5(b, smem, v, nv);
    }
}

int main(int argc, char** argv) {
    const int s1 = argc > 2 ? atoi(argv[1]) : 4, s2 = argc > 2 ? atoi(argv[2]) : 12;
    if ((C0 / s1) % BK || (C1 * 9 / s2) % BK) { fprintf(stderr, "K / splits must be a multiple of %d\n", BK); return 1; }
    const std::string d = "../data/";
    std::vector<half> x(C0 * NP), w1(C1 * C0), w2(C1 * C1 * 9), w3(C0 * C1), out(C0 * NP);
    std::vector<float> b1(C1), b2(C1), b3(C0), yref(C0 * NP);
    read_bin(d + "block_848/x.bin", x);
    read_bin(d + "block_848/y_ref.bin", yref);
    read_bin(d + "Conv_848/w.bin", w1); read_bin(d + "Conv_848/b.bin", b1);
    read_bin(d + "Conv_850/w.bin", w2); read_bin(d + "Conv_850/b.bin", b2);
    read_bin(d + "Conv_852/w.bin", w3); read_bin(d + "Conv_852/b.bin", b3);

    auto up = [](const auto& v) {
        using T = typename std::decay_t<decltype(v)>::value_type;
        T* p; CUDA_CHECK(cudaMalloc(&p, v.size() * sizeof(T)));
        CUDA_CHECK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
        return (const T*)p;
    };
    auto zeros = [](size_t n, auto* tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        T* p; CUDA_CHECK(cudaMalloc(&p, n * sizeof(T))); CUDA_CHECK(cudaMemset(p, 0, n * sizeof(T)));
        return p;
    };
    Block b{up(x), up(w1), up(w2), up(w3), up(b1), up(b2), up(b3),
            zeros(C0 * NC, (half*)0), zeros((size_t)C1 * 9 * NC, (half*)0), zeros(C1 * NC, (half*)0),
            zeros(C0 * NP, (half*)0), zeros((size_t)s1 * C1 * NC, (float*)0), zeros((size_t)s2 * C1 * NC, (float*)0), s1, s2};

    int sms, per_sm;
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, block_fused, BLK_THREADS, 0));
    const int grid = sms * per_sm;  // every block resident at once, as grid.sync() requires
    void* args[] = {&b};
    auto fused = [&] { CUDA_CHECK(cudaLaunchCooperativeKernel((void*)block_fused, grid, BLK_THREADS, args, 0, 0)); };
    // separate launches: GEMM stages get one block per tile, elementwise stages enough blocks to cover them
    const int nb[6] = {(C0 * NP + BLK_THREADS - 1) / BLK_THREADS, (C1 / 64) * s1, (C1 * NP + BLK_THREADS - 1) / BLK_THREADS,
                       (C1 / 64) * s2, (C1 * NP + BLK_THREADS - 1) / BLK_THREADS, C0 / 64};
    auto separate = [&] { for (int st = 0; st < 6; ++st) k_stage<<<nb[st], BLK_THREADS>>>(b, st); };

    auto check = [&](const char* name) {
        CUDA_CHECK(cudaMemset(b.out, 0, C0 * NP * sizeof(half)));
        return name;
    };
    double mag = 0.0;
    for (float v : yref) mag = fmax(mag, fabs(v));
    auto err = [&] {
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(out.data(), b.out, out.size() * sizeof(half), cudaMemcpyDeviceToHost));
        double e = 0.0;
        for (size_t i = 0; i < out.size(); ++i) {
            double dd = fabs(__half2float(out[i]) - yref[i]);
            if (!(dd <= e)) e = dd;  // also catches NaN
        }
        return e;
    };
    check("fused"); fused(); double e_f = err();
    check("separate"); separate(); double e_s = err();
    float t_f = time_ms(fused), t_s = time_ms(separate);
    bool ok = e_f <= 0.01 * mag && e_s <= 0.01 * mag;
    printf("block_848 (Conv_848 -> Conv_850 -> Conv_852 + residual), splits %d/%d, grid %d blocks (%d SMs x %d)\n", s1, s2,
           grid, sms, per_sm);
    printf("  fused (1 cooperative launch)   %8.4f ms   max|out - onnx| %.4g\n", t_f, e_f);
    printf("  same stages, 6 launches        %8.4f ms   max|out - onnx| %.4g   (max|onnx| %.2f)\n", t_s, e_s, mag);
    printf("RESULT s1=%d s2=%d fused_ms=%.5f separate_ms=%.5f err=%.5g ok=%d\n", s1, s2, t_f, t_s, fmax(e_f, e_s), ok);
    return ok ? 0 : 2;
}
