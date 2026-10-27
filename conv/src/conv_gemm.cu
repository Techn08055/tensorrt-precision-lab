// conv_gemm.cu
// Any ResNet50 conv layer (square input, symmetric pad, group 1) as im2col + GEMM, batch 1, FP16,
// using the best kernel from the per-layer work: 2-3 stage cp.async pipeline, padded shared-memory
// rows, split-K with FP32 partials, then one reduce that adds bias (+ ReLU if the layer has one).
//
// As a GEMM: M = Cout, N = Ho*Wo (zero-padded to a multiple of 64), K = Cin*k*k (zero-padded to a
// multiple of 32; only the 7x7 stem needs it, 147 -> 160). k = ci*k*k + kh*k + kw, which is the
// ONNX weight layout, so W is used as-is apart from the K padding (done once, like a build step).
// The split count and pipeline depth are picked per layer by timing every candidate.
//
// Build: nvcc -O3 -arch=sm_89 conv_gemm.cu -o conv_gemm -lcublas
// Run:   ./conv_gemm ../data/Conv_850     (tensors + meta.txt from ../benchmarks/export_conv_real.py)
// Last line of output is "RESULT key=value ..." for ../benchmarks/run_all_layers.py.

// Reuse the pipelined split-K kernel, timer, file reader and cuBLAS handle.
#define CONV846_NO_MAIN
#include "conv846_gemm.cu"

// Xc[c*kk + t][p] = X[c][oh*s + kh - pad][ow*s + kw - pad]; zero outside the input and for p >= n.
// One thread per (input channel, output pixel) writes all k*k taps. Rows k >= Cin*k*k (K padding)
// are never written and stay zero from the memset at startup.
__global__ void im2col_any(const half* X, half* Xc, int cin, int h, int k, int s, int pad, int ho, int npad) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int c = i / npad, p = i % npad;
    if (c >= cin) return;
    int oh = (p / ho) * s - pad, ow = (p % ho) * s - pad;
    bool valid_p = p < ho * ho;
    for (int t = 0; t < k * k; ++t) {
        int ih = oh + t / k, iw = ow + t % k;
        bool in = valid_p && ih >= 0 && ih < h && iw >= 0 && iw < h;
        Xc[(size_t)(c * k * k + t) * npad + p] = in ? X[((size_t)c * h + ih) * h + iw] : __float2half(0.0f);
    }
}

// sum split-K partials, add bias, optional ReLU, drop padding columns -> NCHW Cout x Ho x Wo
__global__ void reduce_any(const float* P, const float* bias, half* Y, int splits, int m, int n, int npad, bool relu) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m * n) return;
    int co = i / n, p = i % n;
    float s = bias[co];
    for (int z = 0; z < splits; ++z) s += P[((size_t)z * m + co) * npad + p];
    Y[i] = __float2half(relu ? fmaxf(s, 0.0f) : s);
}

// One GEMM launch for the chosen pipeline depth, copy width and epilogue.
// KS = 3: implicit GEMM straight from X (the conv geometry after relu is only used then).
// epi: one split, write the final output from the GEMM; otherwise FP32 partials for the reduce kernel.
template <int ST, int BV, int KS = 0>
static void pipe(dim3 g, bool epi, const half* W, const half* B, float* P, int m, int npad, int kpad, int ldb, int n,
                 half* Y, const float* bias, bool relu, int cin = 0, int h = 0, int s = 1, int pad = 0, int ho = 1,
                 int kr = 0) {
    if (epi) gemm_wmma_splitk_pipe<ST, float, 8, BV, true, KS><<<g, BLK_THREADS>>>(W, B, P, m, npad, kpad, ldb, n, Y,
                                                                                   bias, relu, cin, h, s, pad, ho, kr);
    else gemm_wmma_splitk_pipe<ST, float, 8, BV, false, KS><<<g, BLK_THREADS>>>(W, B, P, m, npad, kpad, ldb, n, nullptr,
                                                                                   nullptr, false, cin, h, s, pad, ho, kr);
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s ../data/Conv_850 [nodirect] [noepi] [noimplicit]\n", argv[0]); return 1; }
    std::string dir = argv[1];
    // switches to measure each change separately: old padded copy for 1x1 s1, old partials + reduce for 1 split
    bool allow_direct = true, allow_epi = true, allow_implicit = true;
    for (int i = 2; i < argc; ++i) {
        if (std::string(argv[i]) == "noimplicit") allow_implicit = false;
        if (std::string(argv[i]) == "nodirect") allow_direct = false;
        if (std::string(argv[i]) == "noepi") allow_epi = false;
    }
    std::string layer = dir.substr(dir.find_last_of('/') + 1);
    int cin, h, cout, k, s, pad, ho, relu;
    FILE* mf = fopen((dir + "/meta.txt").c_str(), "r");
    if (!mf || fscanf(mf, "%d %d %d %d %d %d %d %d", &cin, &h, &cout, &k, &s, &pad, &ho, &relu) != 8) {
        fprintf(stderr, "cannot read %s/meta.txt (run export_conv_real.py)\n", dir.c_str());
        return 1;
    }
    fclose(mf);
    const int m = cout, n = ho * ho, npad = (n + BN - 1) / BN * BN;
    const int kr = cin * k * k, kpad = (kr + BK - 1) / BK * BK;
    if (m % BM != 0) { fprintf(stderr, "Cout %d not a multiple of %d\n", m, BM); return 1; }

    std::vector<half> hX((size_t)cin * h * h), hWr((size_t)m * kr), hW((size_t)m * kpad, __float2half(0.0f));
    std::vector<half> hY((size_t)m * n);
    std::vector<float> hB(m), hYref((size_t)m * n);
    read_bin(dir + "/x.bin", hX);
    read_bin(dir + "/w.bin", hWr);
    read_bin(dir + "/b.bin", hB);
    read_bin(dir + "/y_ref.bin", hYref);
    for (int r = 0; r < m; ++r) memcpy(&hW[(size_t)r * kpad], &hWr[(size_t)r * kr], kr * sizeof(half));
    if (relu) for (float& v : hYref) v = fmaxf(v, 0.0f);
    double mag = 0.0;
    for (float v : hYref) mag = fmax(mag, fabs(v));

    // split-K candidates: (kpad/splits) a multiple of BK, partial buffer capped at 64 MB
    std::vector<int> split_opts;
    for (int sp : {1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 32, 36, 48})
        if ((kpad / BK) % sp == 0 && (size_t)sp * m * npad * sizeof(float) <= ((size_t)64 << 20)) split_opts.push_back(sp);
    const int max_splits = split_opts.back();

    half *dX, *dXc, *dW, *dY;
    float *dP, *dBias;
    CUDA_CHECK(cudaMalloc(&dX, hX.size() * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dXc, (size_t)kpad * npad * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dW, hW.size() * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dY, hY.size() * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dP, (size_t)max_splits * m * npad * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dBias, m * sizeof(float)));
    CUDA_CHECK(cudaMemset(dXc, 0, (size_t)kpad * npad * sizeof(half)));  // K-padding rows stay zero
    CUDA_CHECK(cudaMemcpy(dX, hX.data(), hX.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dW, hW.data(), hW.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBias, hB.data(), m * sizeof(float), cudaMemcpyHostToDevice));
    if (cublasCreate(&g_handle) != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cublasCreate failed\n"); return 1; }

    // A 1x1 stride-1 conv needs no im2col: X itself is the K x N matrix (row stride n). The GEMM reads it
    // in place and zero-fills the columns past n, so no padded copy. Needs n % 4 == 0 for 8-byte copies.
    // shortcut: n = 49 (7x7) still takes the padded copy, its rows aren't 4-half aligned; 2-byte loads would fix it.
    const bool direct = allow_direct && k == 1 && s == 1 && pad == 0 && n % 4 == 0;
    const half* dB = direct ? dX : dXc;
    const int ldb = direct ? n : npad, bvec = direct && n % 8 != 0 ? 4 : 8;
    auto im2col = [&] {
        if (direct) return;
        int threads = cin * npad;
        im2col_any<<<(threads + 255) / 256, 256>>>(dX, dXc, cin, h, k, s, pad, ho, npad);
    };
    auto gemm = [&](int stages, int splits, bool implicit = false) {
        dim3 g(npad / BN, m / BM, splits);
        bool epi = allow_epi && splits == 1;  // one split: write the final output from the GEMM itself
        if (implicit && stages == 2) pipe<2, 8, 3>(g, epi, dW, dX, dP, m, npad, kpad, 0, n, dY, dBias, relu, cin, h, s, pad, ho, kr);
        else if (implicit) pipe<3, 8, 3>(g, epi, dW, dX, dP, m, npad, kpad, 0, n, dY, dBias, relu, cin, h, s, pad, ho, kr);
        else if (stages == 2 && bvec == 8) pipe<2, 8>(g, epi, dW, dB, dP, m, npad, kpad, ldb, n, dY, dBias, relu);
        else if (stages == 2) pipe<2, 4>(g, epi, dW, dB, dP, m, npad, kpad, ldb, n, dY, dBias, relu);
        else if (bvec == 8) pipe<3, 8>(g, epi, dW, dB, dP, m, npad, kpad, ldb, n, dY, dBias, relu);
        else pipe<3, 4>(g, epi, dW, dB, dP, m, npad, kpad, ldb, n, dY, dBias, relu);
        if (!epi) reduce_any<<<(m * n + 255) / 256, 256>>>(dP, dBias, dY, splits, m, n, npad, relu);
    };
    auto cublas = [&] {
        const float alpha = 1.0f, beta = 0.0f;
        // row-major Y(m x n) = W * Xc[:, :n]; column-major: Y^T = Xc^T * W^T, Xc rows have stride npad
        cublasStatus_t st = cublasGemmEx(g_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, kpad, &alpha, dB, CUDA_R_16F, ldb,
                                         dW, CUDA_R_16F, kpad, &beta, dY, CUDA_R_16F, n, CUBLAS_COMPUTE_32F,
                                         CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        if (st != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cublasGemmEx failed: %d\n", (int)st); exit(1); }
    };
    // max |dY (+ bias, ReLU if this path didn't apply them) - ONNX output (already ReLU'd if relu)|
    auto err = [&](bool epilogue_on_host) {
        CUDA_CHECK(cudaMemcpy(hY.data(), dY, hY.size() * sizeof(half), cudaMemcpyDeviceToHost));
        double e = 0.0;
        for (size_t i = 0; i < hY.size(); ++i) {
            double v = __half2float(hY[i]);
            if (epilogue_on_host) { v += hB[i / n]; if (relu) v = fmax(v, 0.0); }
            double d = fabs(v - hYref[i]);
            if (!(d <= e)) e = d;  // also catches NaN
        }
        return e;
    };

    const double flops = 2.0 * m * n * kr;  // useful work only (no padding)
    printf("%s: %dx%dx%d -> %dx%dx%d k%d s%d p%d relu%d | GEMM M=%d N=%d(pad %d) K=%d(pad %d)%s\n", layer.c_str(), cin, h, h,
           cout, ho, ho, k, s, pad, relu, m, n, npad, kr, kpad, direct ? " | no im2col (X is B)" : "");

    im2col();
    cublas();
    // cuBLAS output has no bias/ReLU; checked with them added on the host
    double e_cublas = err(true);
    float t_im2col = time_ms(im2col);
    float t_cublas = time_ms([&] { im2col(); cublas(); });
    printf("  im2col                 %8.4f ms\n", t_im2col);
    printf("  im2col + cublas        %8.4f ms  %8.1f GFLOPS  max|err| %.4g (no bias/ReLU in timing)\n", t_cublas,
           flops / t_cublas / 1e6, e_cublas);

    // explicit im2col and (3x3 layers) implicit GEMM, every split count and depth; fastest wins
    int best_stages = 2, best_splits = 1, best_impl = 0;
    float best_t = 1e30f;
    double best_e = 0.0, worst_e = 0.0;
    for (int impl = 0; impl <= (allow_implicit && k == 3 ? 1 : 0); ++impl)
        for (int stages : {2, 3})
            for (int sp : split_opts) {
                auto run = [&] { if (!impl) im2col(); gemm(stages, sp, impl); };
                CUDA_CHECK(cudaMemset(dY, 0, hY.size() * sizeof(half)));
                run();
                double e = err(false);
                float t = time_ms(run);
                printf("  %s pipe%d split-K %-2d (%5d blocks)  %8.4f ms  %8.1f GFLOPS  %5.2fx vs cublas  max|err| %.4g\n",
                       impl ? "implicit" : "im2col  ", stages, sp, sp * (npad / BN) * (m / BM), t, flops / t / 1e6,
                       t_cublas / t, e);
                worst_e = fmax(worst_e, e);
                if (t < best_t) { best_t = t; best_stages = stages; best_splits = sp; best_e = e; best_impl = impl; }
            }
    bool ok = worst_e <= 0.01 * fmax(mag, 1.0);  // every variant within 1% of the largest output: FP16 rounding level
    printf("RESULT layer=%s cin=%d h=%d cout=%d k=%d s=%d ho=%d relu=%d ours_ms=%.5f stages=%d splits=%d implicit=%d "
           "ours_err=%.5g im2col_ms=%.5f cublas_ms=%.5f cublas_err=%.5g max_ref=%.4g gflop=%.4f ok=%d\n",
           layer.c_str(), cin, h, cout, k, s, ho, relu, best_t, best_stages, best_splits, best_impl, best_e,
           best_impl ? 0.0f : t_im2col, t_cublas, e_cublas, mag, flops / 1e9, ok);
    cublasDestroy(g_handle);
    return ok ? 0 : 2;
}
