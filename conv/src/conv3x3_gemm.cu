// conv3x3_gemm.cu
// The 3x3, pad-1, 512 -> 512 channel convs of ResNet50's last stage, batch 1, FP16:
//   Conv_850, Conv_856  stride 1, 512x7x7  -> 512x7x7
//   Conv_842            stride 2, 512x14x14 -> 512x7x7
// Explicit im2col + split-K WMMA GEMM + reduce (bias, optional ReLU).
//
// As a GEMM: M = 512 output channels, N = 49 output pixels (padded to 64),
// K = 512 input channels x 9 taps = 4608, with k = ci*9 + kh*3 + kw. That k order is
// the ONNX weight layout [co][ci][kh][kw], so W is used as-is as the row-major M x K matrix.
// im2col buffer is 4608 x 64 halves = 590 KB: at batch 1 and 7x7 output the 9x blowup is small.
//
// Build: nvcc -O3 -arch=sm_89 conv3x3_gemm.cu -o conv3x3_gemm -lcublas
// Run:   ./conv3x3_gemm ../data/Conv_850 1    (stride 1; real tensors from ../benchmarks/export_conv_real.py)
//        ./conv3x3_gemm ../data/Conv_842 2

// Reuse the split-K kernel, timer, file reader and cuBLAS handle from the Conv_846 driver.
#define CONV846_NO_MAIN
#include "conv846_gemm.cu"
#include <cstring>

#define C3_CI 512
#define C3_CO 512
#define C3_HW 7  // output size; input is 7*stride
#define C3_K (C3_CI * 9)

// Xc[k][p] = X[ci][oh*s+kh-1][ow*s+kw-1], zero for the padding ring and for columns 49..63
__global__ void im2col_3x3(const half* X, half* Xc, int hin, int s) {
    int p = threadIdx.x, k = blockIdx.x;
    int ci = k / 9, kh = (k % 9) / 3, kw = k % 3;
    int ih = (p / C3_HW) * s + kh - 1, iw = (p % C3_HW) * s + kw - 1;
    bool in = p < NPIX && ih >= 0 && ih < hin && iw >= 0 && iw < hin;
    Xc[k * NPAD + p] = in ? X[(ci * hin + ih) * hin + iw] : __float2half(0.0f);
}

// v2: one thread per (ci, pixel) writes all 9 taps: 128 blocks of 256 instead of 4608 of 64.
// Stores stay coalesced (consecutive threads -> consecutive pixels of one k row).
__global__ void im2col_3x3_v2(const half* X, half* Xc, int hin, int s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int ci = i / NPAD, p = i % NPAD;
    if (ci >= C3_CI) return;
    int oh = (p / C3_HW) * s, ow = (p % C3_HW) * s;
    #pragma unroll
    for (int t = 0; t < 9; ++t) {
        int ih = oh + t / 3 - 1, iw = ow + t % 3 - 1;
        bool in = p < NPIX && ih >= 0 && ih < hin && iw >= 0 && iw < hin;
        Xc[(ci * 9 + t) * NPAD + p] = in ? X[(ci * hin + ih) * hin + iw] : __float2half(0.0f);
    }
}

// sum split-K partials, add bias, optional ReLU, drop padding columns -> NCHW 512x7x7
template <typename AccT>
__global__ void reduce_bias_c3(const AccT* P, const float* bias, half* Y, int splits, bool relu) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= C3_CO * NPIX) return;
    int co = i / NPIX, p = i % NPIX;
    float s = bias[co];
    for (int z = 0; z < splits; ++z) s += (float)P[((size_t)z * C3_CO + co) * NPAD + p];
    Y[i] = __float2half(relu ? fmaxf(s, 0.0f) : s);
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s ../data/Conv_850 [stride]\n", argv[0]); return 1; }
    std::string dir = argv[1];
    const int stride = argc > 2 ? atoi(argv[2]) : 1;
    const int hin = C3_HW * stride;
    std::string layer = dir.substr(dir.find_last_of('/') + 1);  // e.g. Conv_850
    // TensorRT FP16 per-layer profile, each fused with its ReLU (../../profiling/PROFILE_RESULTS.md)
    const double trt_fp16_ms = layer == "Conv_842" ? 0.043 : 0.038;

    std::vector<half> hX((size_t)C3_CI * hin * hin), hW((size_t)C3_CO * C3_K), hY((size_t)C3_CO * NPIX);
    std::vector<float> hB(C3_CO), hYref((size_t)C3_CO * NPIX);
    read_bin(dir + "/x.bin", hX);
    read_bin(dir + "/w.bin", hW);
    read_bin(dir + "/b.bin", hB);
    read_bin(dir + "/y_ref.bin", hYref);

    const int max_splits = 48;
    half *dX, *dXc, *dW, *dY;
    float *dP, *dBias;
    CUDA_CHECK(cudaMalloc(&dX, hX.size() * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dXc, (size_t)C3_K * NPAD * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dW, hW.size() * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dY, hY.size() * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dP, (size_t)max_splits * C3_CO * NPAD * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dBias, C3_CO * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dX, hX.data(), hX.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dW, hW.data(), hW.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBias, hB.data(), C3_CO * sizeof(float), cudaMemcpyHostToDevice));
    if (cublasCreate(&g_handle) != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cublasCreate failed\n"); return 1; }

    auto im2col = [&] { im2col_3x3<<<C3_K, NPAD>>>(dX, dXc, hin, stride); };
    auto im2col_v2 = [&] { im2col_3x3_v2<<<C3_CI * NPAD / 256, 256>>>(dX, dXc, hin, stride); };
    auto reduce = [&](int splits, bool relu) {
        reduce_bias_c3<float><<<(C3_CO * NPIX + 255) / 256, 256>>>(dP, dBias, dY, splits, relu);
    };
    // same split-K GEMM with FP16 accumulators and FP16 partials (summed in FP32 by the reduce)
    half* dP16 = (half*)dP;  // dP is sized for float partials, so half partials fit
    auto splitk16 = [&](int splits, bool relu = false) {
        gemm_wmma_splitk<0, half><<<dim3(NPAD / BN, C3_CO / BM, splits), dim3(BLK_THREADS)>>>(dW, dXc, dP16, C3_CO, NPAD, C3_K);
        reduce_bias_c3<half><<<(C3_CO * NPIX + 255) / 256, 256>>>(dP16, dBias, dY, splits, relu);
    };
    auto splitk = [&](int splits, bool relu = false) {
        gemm_wmma_splitk<0><<<dim3(NPAD / BN, C3_CO / BM, splits), dim3(BLK_THREADS)>>>(dW, dXc, dP, C3_CO, NPAD, C3_K);
        reduce(splits, relu);
    };
    // pipelined split-K: STAGES-deep cp.async, FP32 or FP16 accumulators
    auto splitk_pipe = [&](int stages, bool fp16, int splits, bool relu = false) {
        dim3 g(NPAD / BN, C3_CO / BM, splits), b(BLK_THREADS);
        if (fp16) {
            if (stages == 2) gemm_wmma_splitk_pipe<2, half><<<g, b>>>(dW, dXc, dP16, C3_CO, NPAD, C3_K);
            else if (stages == 3) gemm_wmma_splitk_pipe<3, half><<<g, b>>>(dW, dXc, dP16, C3_CO, NPAD, C3_K);
            else gemm_wmma_splitk_pipe<4, half><<<g, b>>>(dW, dXc, dP16, C3_CO, NPAD, C3_K);
            reduce_bias_c3<half><<<(C3_CO * NPIX + 255) / 256, 256>>>(dP16, dBias, dY, splits, relu);
        } else {
            if (stages == 2) gemm_wmma_splitk_pipe<2><<<g, b>>>(dW, dXc, dP, C3_CO, NPAD, C3_K);
            else if (stages == 3) gemm_wmma_splitk_pipe<3><<<g, b>>>(dW, dXc, dP, C3_CO, NPAD, C3_K);
            else gemm_wmma_splitk_pipe<4><<<g, b>>>(dW, dXc, dP, C3_CO, NPAD, C3_K);
            reduce_bias_c3<float><<<(C3_CO * NPIX + 255) / 256, 256>>>(dP, dBias, dY, splits, relu);
        }
    };
    // implicit GEMM: im2col happens inside the B tile load, reads raw X.
    // shortcut: load mode 2 is written for 7x7 stride-1 input only, so it's skipped for Conv_842.
    auto implicit = [&](int splits) {
        gemm_wmma_splitk<2><<<dim3(NPAD / BN, C3_CO / BM, splits), dim3(BLK_THREADS)>>>(dW, dX, dP, C3_CO, NPAD, C3_K);
        reduce(splits, false);
    };
    auto cublas = [&] {
        const float alpha = 1.0f, beta = 0.0f;
        // row-major Y(512x49) = W * Xc[:, :49]; column-major: Y^T = Xc^T * W^T, Xc rows have stride 64
        cublasStatus_t st = cublasGemmEx(g_handle, CUBLAS_OP_N, CUBLAS_OP_N, NPIX, C3_CO, C3_K, &alpha, dXc,
                                         CUDA_R_16F, NPAD, dW, CUDA_R_16F, C3_K, &beta, dY, CUDA_R_16F, NPIX,
                                         CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        if (st != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cublasGemmEx failed: %d\n", (int)st); exit(1); }
    };
    // same GEMM with FP16 accumulation, to test whether that's where TensorRT's speed comes from
    auto cublas16 = [&] {
        const half alpha = __float2half(1.0f), beta = __float2half(0.0f);
        cublasStatus_t st = cublasGemmEx(g_handle, CUBLAS_OP_N, CUBLAS_OP_N, NPIX, C3_CO, C3_K, &alpha, dXc,
                                         CUDA_R_16F, NPAD, dW, CUDA_R_16F, C3_K, &beta, dY, CUDA_R_16F, NPIX,
                                         CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        if (st != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cublasGemmEx 16F failed: %d\n", (int)st); exit(1); }
    };
    // max |dY (+ bias if the kernel didn't add it) - ONNX fp32 output (ReLU'd if asked)|
    auto err = [&](bool add_bias, bool relu = false) {
        CUDA_CHECK(cudaMemcpy(hY.data(), dY, hY.size() * sizeof(half), cudaMemcpyDeviceToHost));
        double m = 0.0;
        for (size_t i = 0; i < hY.size(); ++i) {
            double r = relu ? fmax(hYref[i], 0.0f) : hYref[i];
            double d = fabs(__half2float(hY[i]) + (add_bias ? hB[i / NPIX] : 0.0f) - r);
            if (!(d <= m)) m = d;  // also catches NaN
        }
        return m;
    };
    double mag = 0.0;
    for (float v : hYref) mag = fmax(mag, fabs(v));

    printf("%s: 3x3 conv s%d p1, %dx%dx%d -> %dx%dx%d, as GEMM M=%d N=%d(pad %d) K=%d\n", layer.c_str(), stride,
           C3_CI, hin, hin, C3_CO, C3_HW, C3_HW, C3_CO, NPIX, NPAD, C3_K);
    const double flops = 2.0 * C3_CO * NPIX * C3_K;

    im2col();
    cublas();
    double e_cublas = err(true);
    float t_im2col = time_ms(im2col);
    float t_cublas = time_ms([&] { im2col_v2(); cublas(); });  // fastest im2col, same as our best path
    printf("  im2col only                     %8.4f ms\n", t_im2col);
    printf("  im2col_v2 + cublas (no bias)    %8.4f ms  %8.1f GFLOPS  max|err| %.4g (max|onnx| %.2f)\n", t_cublas,
           flops / t_cublas / 1e6, e_cublas, mag);
    im2col_v2();
    cublas16();
    double e_cublas16 = err(true);
    float t_cublas16 = time_ms([&] { im2col_v2(); cublas16(); });
    printf("  im2col_v2 + cublas FP16-accum   %8.4f ms  %8.1f GFLOPS  max|err| %.4g\n", t_cublas16,
           flops / t_cublas16 / 1e6, e_cublas16);
    printf("  TensorRT FP16 layer (+bias+ReLU) %7.4f ms  %8.1f GFLOPS  (per-layer profile)\n", trt_fp16_ms,
           flops / trt_fp16_ms / 1e6);

    std::string id = layer.substr(layer.find('_') + 1);  // "850"
    FILE* csv = fopen(("../results/conv" + id + "_gemm_real.csv").c_str(), "w");
    if (!csv) csv = fopen(("conv" + id + "_gemm_real.csv").c_str(), "w");
    fprintf(csv, "variant,mean_ms,gflops,max_abs_vs_onnx\n");
    fprintf(csv, "im2col,%.5f,,\n", t_im2col);
    fprintf(csv, "im2col_v2_plus_cublas,%.5f,%.1f,%.6g\n", t_cublas, flops / t_cublas / 1e6, e_cublas);
    fprintf(csv, "im2col_v2_plus_cublas_fp16accum,%.5f,%.1f,%.6g\n", t_cublas16, flops / t_cublas16 / 1e6, e_cublas16);
    fprintf(csv, "tensorrt_fp16_profile_with_relu,%.5f,%.1f,\n", trt_fp16_ms, flops / trt_fp16_ms / 1e6);

    // v2 im2col must produce the identical buffer
    std::vector<half> c1((size_t)C3_K * NPAD), c2(c1.size());
    im2col();
    CUDA_CHECK(cudaMemcpy(c1.data(), dXc, c1.size() * sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemset(dXc, 0xFF, c1.size() * sizeof(half)));
    im2col_v2();
    CUDA_CHECK(cudaMemcpy(c2.data(), dXc, c2.size() * sizeof(half), cudaMemcpyDeviceToHost));
    if (memcmp(c1.data(), c2.data(), c1.size() * sizeof(half)) != 0) { fprintf(stderr, "im2col_v2 mismatch\n"); return 1; }
    float t_im2col_v2 = time_ms(im2col_v2);
    printf("  im2col v2 only (identical buffer) %6.4f ms\n", t_im2col_v2);
    fprintf(csv, "im2col_v2,%.5f,,\n", t_im2col_v2);

    // split counts with (4608/splits) % 32 == 0
    const int split_opts[] = {1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 36, 48};
    const char* vname[4] = {"im2col", "im2col_v2", "implicit", "v2_fp16acc"};
    int best_splits = 1;
    float best_t = 1e30f;
    int best16_splits = 1;
    float best16_t = 1e30f;
    for (int v = 0; v < 4; ++v) {
        if (v == 2 && stride != 1) continue;
        for (int splits : split_opts) {
            auto run = [&] {
                if (v == 0) { im2col(); splitk(splits); }
                else if (v == 1) { im2col_v2(); splitk(splits); }
                else if (v == 2) implicit(splits);
                else { im2col_v2(); splitk16(splits); }
            };
            CUDA_CHECK(cudaMemset(dY, 0, hY.size() * sizeof(half)));
            run();
            double e = err(false);
            float t = time_ms(run);
            if (v == 1 && t < best_t) { best_t = t; best_splits = splits; }
            if (v == 3 && t < best16_t) { best16_t = t; best16_splits = splits; }
            printf("  %-9s + split-K %-2d (%3d blocks) + reduce/bias %8.4f ms  %8.1f GFLOPS  %5.2fx vs cublas path  max|err| %.4g\n",
                   vname[v], splits, splits * C3_CO / BM, t, flops / t / 1e6, t_cublas / t, e);
            fprintf(csv, "%s_splitk%d_bias,%.5f,%.1f,%.6g\n", vname[v], splits, t, flops / t / 1e6, e);
        }
    }

    // best path with ReLU fused into the reduce: the same work TensorRT's fused layer does
    CUDA_CHECK(cudaMemset(dY, 0, hY.size() * sizeof(half)));
    im2col_v2();
    splitk(best_splits, true);
    double e_relu = err(false, true);
    float t_relu = time_ms([&] { im2col_v2(); splitk(best_splits, true); });
    printf("  best: im2col_v2 + split-K %d + reduce/bias/ReLU %8.4f ms  %8.1f GFLOPS  max|err vs relu(onnx)| %.4g\n",
           best_splits, t_relu, flops / t_relu / 1e6, e_relu);
    fprintf(csv, "im2col_v2_splitk%d_bias_relu,%.5f,%.1f,%.6g\n", best_splits, t_relu, flops / t_relu / 1e6, e_relu);
    // pipelined + padded shared-memory rows (PAD = 8 halves), FP32 or FP16 accumulators
    auto splitk_pipe_pad = [&](int stages, int splits, bool fp16, bool relu = false) {
        dim3 g(NPAD / BN, C3_CO / BM, splits), b(BLK_THREADS);
        if (fp16) {
            if (stages == 2) gemm_wmma_splitk_pipe<2, half, 8><<<g, b>>>(dW, dXc, dP16, C3_CO, NPAD, C3_K);
            else if (stages == 3) gemm_wmma_splitk_pipe<3, half, 8><<<g, b>>>(dW, dXc, dP16, C3_CO, NPAD, C3_K);
            else gemm_wmma_splitk_pipe<4, half, 8><<<g, b>>>(dW, dXc, dP16, C3_CO, NPAD, C3_K);
            reduce_bias_c3<half><<<(C3_CO * NPIX + 255) / 256, 256>>>(dP16, dBias, dY, splits, relu);
            return;
        }
        if (stages == 2) gemm_wmma_splitk_pipe<2, float, 8><<<g, b>>>(dW, dXc, dP, C3_CO, NPAD, C3_K);
        else if (stages == 3) gemm_wmma_splitk_pipe<3, float, 8><<<g, b>>>(dW, dXc, dP, C3_CO, NPAD, C3_K);
        else gemm_wmma_splitk_pipe<4, float, 8><<<g, b>>>(dW, dXc, dP, C3_CO, NPAD, C3_K);
        reduce_bias_c3<float><<<(C3_CO * NPIX + 255) / 256, 256>>>(dP, dBias, dY, splits, relu);
    };
    int bestp_stages = 2, bestp_splits = 12, bestp_fp16 = 0;
    float bestp_t = 1e30f;
    for (int fp16 = 0; fp16 < 2; ++fp16)
    for (int stages = 2; stages <= 4; ++stages)
        for (int splits : {1, 4, 6, 8, 12, 16, 24}) {
            auto run = [&] { im2col_v2(); splitk_pipe_pad(stages, splits, fp16); };
            CUDA_CHECK(cudaMemset(dY, 0, hY.size() * sizeof(half)));
            run();
            double e = err(false);
            float t = time_ms(run);
            if (!fp16 && t < bestp_t) { bestp_t = t; bestp_stages = stages; bestp_splits = splits; }
            printf("  pipe%d pad8 %s + split-K %-2d (%3d blocks) + reduce/bias %8.4f ms  %8.1f GFLOPS  %5.2fx vs cublas path  max|err| %.4g\n",
                   stages, fp16 ? "fp16acc" : "fp32acc", splits, splits * C3_CO / BM, t, flops / t / 1e6, t_cublas / t, e);
            fprintf(csv, "im2col_v2_pipe%d_pad8_%s_splitk%d_bias,%.5f,%.1f,%.6g\n", stages, fp16 ? "fp16acc" : "fp32acc",
                    splits, t, flops / t / 1e6, e);
        }
    {   // best pipelined + padded FP32-acc config, with ReLU: the like-for-like number against TensorRT
        CUDA_CHECK(cudaMemset(dY, 0, hY.size() * sizeof(half)));
        im2col_v2();
        splitk_pipe_pad(bestp_stages, bestp_splits, false, true);
        double e = err(false, true);
        float t = time_ms([&] { im2col_v2(); splitk_pipe_pad(bestp_stages, bestp_splits, false, true); });
        printf("  best pipelined: im2col_v2 + pipe%d pad8 fp32acc split-K %d + reduce/bias/ReLU %8.4f ms  %8.1f GFLOPS  max|err vs relu(onnx)| %.4g\n",
               bestp_stages, bestp_splits, t, flops / t / 1e6, e);
        fprintf(csv, "best_im2col_v2_pipe%d_pad8_fp32acc_splitk%d_bias_relu,%.5f,%.1f,%.6g\n", bestp_stages, bestp_splits, t,
                flops / t / 1e6, e);
    }

    // pipelined variants: stages x accumulator type x split count
    for (int fp16 = 0; fp16 < 2; ++fp16)
        for (int stages = 2; stages <= 4; ++stages)
            for (int splits : {1, 4, 6, 8, 12, 16, 24}) {
                auto run = [&] { im2col_v2(); splitk_pipe(stages, fp16, splits); };
                CUDA_CHECK(cudaMemset(dY, 0, hY.size() * sizeof(half)));
                run();
                double e = err(false);
                float t = time_ms(run);
                printf("  pipe%d %s + split-K %-2d (%3d blocks) + reduce/bias %8.4f ms  %8.1f GFLOPS  %5.2fx vs cublas path  max|err| %.4g\n",
                       stages, fp16 ? "fp16acc" : "fp32acc", splits, splits * C3_CO / BM, t, flops / t / 1e6, t_cublas / t, e);
                fprintf(csv, "im2col_v2_pipe%d_%s_splitk%d_bias,%.5f,%.1f,%.6g\n", stages, fp16 ? "fp16acc" : "fp32acc",
                        splits, t, flops / t / 1e6, e);
            }

    CUDA_CHECK(cudaMemset(dY, 0, hY.size() * sizeof(half)));
    im2col_v2();
    splitk16(best16_splits, true);
    double e16_relu = err(false, true);
    float t16_relu = time_ms([&] { im2col_v2(); splitk16(best16_splits, true); });
    printf("  best FP16-acc: im2col_v2 + split-K %d + reduce/bias/ReLU %8.4f ms  %8.1f GFLOPS  max|err vs relu(onnx)| %.4g\n",
           best16_splits, t16_relu, flops / t16_relu / 1e6, e16_relu);
    fprintf(csv, "im2col_v2_splitk%d_fp16acc_bias_relu,%.5f,%.1f,%.6g\n", best16_splits, t16_relu, flops / t16_relu / 1e6, e16_relu);
    fclose(csv);
    cublasDestroy(g_handle);
    return 0;
}
