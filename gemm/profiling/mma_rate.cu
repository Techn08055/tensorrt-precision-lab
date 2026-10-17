// mma_rate.cu
// Raw WMMA tensor-core throughput on this GPU, FP16 inputs, FP32 vs FP16 accumulators.
// No memory traffic: fragments are filled once, then each warp runs a long chain of
// mma_sync on 8 independent accumulators (so one mma never waits on the previous one).
// Swept over block count (few blocks = what TensorRT's conv kernels launch) and warps/block.
//
// Build: nvcc -O3 -arch=sm_89 mma_rate.cu -o mma_rate
// Run:   ./mma_rate

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>

using namespace nvcuda;

#define NACC 8
#define ITERS 4096

template <typename AccT>
__global__ void mma_loop(float* out) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, AccT> acc[NACC];
    wmma::fill_fragment(a, __float2half(0.001f));
    wmma::fill_fragment(b, __float2half(0.001f));
    #pragma unroll
    for (int i = 0; i < NACC; ++i) wmma::fill_fragment(acc[i], (AccT)0.0f);
    for (int it = 0; it < ITERS; ++it) {
        #pragma unroll
        for (int i = 0; i < NACC; ++i) wmma::mma_sync(acc[i], a, b, acc[i]);
    }
    float s = 0.0f;  // keep the result live so the loop isn't removed
    #pragma unroll
    for (int i = 0; i < NACC; ++i) s += (float)acc[i].x[0];
    if (s == 12345.0f) out[0] = s;
}

template <typename AccT>
static double tflops(int blocks, int warps) {
    float* d;
    cudaMalloc(&d, sizeof(float));
    mma_loop<AccT><<<blocks, warps * 32>>>(d);  // warmup
    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);
    cudaEventRecord(s);
    const int reps = 5;
    for (int r = 0; r < reps; ++r) mma_loop<AccT><<<blocks, warps * 32>>>(d);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms;
    cudaEventElapsedTime(&ms, s, e);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { fprintf(stderr, "%s\n", cudaGetErrorString(err)); return -1; }
    cudaFree(d);
    double flop = 2.0 * 16 * 16 * 16 * NACC * (double)ITERS * warps * blocks * reps;
    return flop / (ms / 1e3) / 1e12;
}

int main() {
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("%s, %d SMs\n", p.name, p.multiProcessorCount);
    printf("blocks warps/blk |  FP32-acc TFLOPS  FP16-acc TFLOPS  ratio\n");
    const int blocks_list[] = {8, 16, 20, 40, 80, 160};
    const int warps_list[] = {4, 8};
    for (int w : warps_list)
        for (int b : blocks_list) {
            double f32 = tflops<float>(b, w), f16 = tflops<half>(b, w);
            printf("%6d %9d |  %15.1f  %15.1f  %5.2fx\n", b, w, f32, f16, f16 / f32);
        }
    return 0;
}
