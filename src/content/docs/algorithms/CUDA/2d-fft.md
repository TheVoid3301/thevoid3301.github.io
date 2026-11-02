---
title: "2D FFT"
description: "2D FFT"
date: 2026-10-22
tags:
    - Algorithm
    - CUDA
    - GPU
sidebar:
    order: 9
---

# 2D FFT

> **题目来源**：[2D FFT](https://leetgpu.com/challenges/2d-fft) · **难度**：Medium · **语言**：CUDA C++  

## 题目描述

Implement a GPU program to compute the 2D Discrete Fourier Transform (2D DFT) of a complex-valued signal stored on the GPU.

Given a 2D complex-valued input signal of shape (M \times N), compute its 2D DFT using a row-column decomposition:

Apply a 1D DFT to each row of the input signal.

Apply a 1D DFT to each column of the intermediate result.

All values are represented using 32-bit floating-point numbers.

Input and Output Format

The input and output are stored in 1D arrays using an interleaved real-imaginary representation in row-major order.

For an element (x[m,n]):

Real part is stored at index:

$$
2 \cdot (m \cdot N + n)
$$

Imaginary part is stored at index:

$$
2 \cdot (m \cdot N + n) + 1
$$

The 2D DFT is mathematically defined as:

$$
X[u,v] =
\sum_{m=0}^{M-1}
\sum_{n=0}^{N-1}
x[m,n],
e^{-j2\pi\left(\frac{um}{M}+\frac{vn}{N}\right)}
$$

where (X[u,v]) is the resulting frequency-domain signal.

Implementation Requirements

Use only native features. External libraries are not permitted.

The solve function signature must remain unchanged.

The final result must be stored in spectrum.

Example

Input:

M = 2, N = 2

Signal (Real):
[[1, 0],
 [0, 0]]

Signal (Imaginary):
[[0, 0],
 [0, 0]]

Output:

Spectrum (Real):
[[1, 1],
 [1, 1]]

Spectrum (Imaginary):
[[0, 0],
 [0, 0]]

Constraints

(1 \leq M, N \leq 4096)

All signal values are 32-bit floating-point numbers (real and imaginary components).

Performance is measured with (M = N = 2048).

## Naive

> Naive CUDA Kernel

```cpp

#include <cuda_runtime.h>

constexpr float PI = 3.14159265358979323846f;

// ==========================================
// Kernel 1: Row DFT
// ==========================================

__global__ void row_dft(
    const float* signal,
    float* temp,
    int M,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int total = M * N;

    if (idx >= total) return;

    // 当前线程负责输出 Y[m][k]
    int m = idx / N;
    int k = idx % N;

    float sum_real = 0.0f;
    float sum_imag = 0.0f;

    // 对第 m 行执行 DFT
    for (int n = 0; n < N; ++n) {

        int input_idx = 2 * (m * N + n);

        float xr = signal[input_idx];
        float xi = signal[input_idx + 1];

        float theta = -2.0f * PI * k * n / N;

        float wr = cosf(theta);
        float wi = sinf(theta);

        sum_real += xr * wr - xi * wi;
        sum_imag += xr * wi + xi * wr;
    }

    int output_idx = 2 * (m * N + k);

    temp[output_idx] = sum_real;
    temp[output_idx + 1] = sum_imag;
}


// ==========================================
// Kernel 2: Column DFT
// ==========================================

__global__ void column_dft(
    const float* temp,
    float* spectrum,
    int M,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int total = M * N;

    if (idx >= total) return;

    // 当前线程负责输出 F[p][n]
    int p = idx / N;
    int n = idx % N;

    float sum_real = 0.0f;
    float sum_imag = 0.0f;

    // 对第 n 列执行 DFT
    for (int m = 0; m < M; ++m) {

        int input_idx = 2 * (m * N + n);

        float xr = temp[input_idx];
        float xi = temp[input_idx + 1];

        float theta = -2.0f * PI * p * m / M;

        float wr = cosf(theta);
        float wi = sinf(theta);

        sum_real += xr * wr - xi * wi;
        sum_imag += xr * wi + xi * wr;
    }

    int output_idx = 2 * (p * N + n);

    spectrum[output_idx] = sum_real;
    spectrum[output_idx + 1] = sum_imag;
}


// ==========================================
// Host Function
// ==========================================

extern "C" void solve(
    const float* signal,
    float* spectrum,
    int M,
    int N
) {
    if (M <= 0 || N <= 0) return;

    int threads = 256;
    int total = M * N;

    int blocks = (total + threads - 1) / threads;

    // 中间结果
    float* temp = nullptr;

    cudaMalloc(
        reinterpret_cast<void**>(&temp),
        2ULL * total * sizeof(float)
    );

    // 第一阶段：Row DFT
    row_dft<<<blocks, threads>>>(
        signal, temp, M, N
    );

    // 第二阶段：Column DFT
    column_dft<<<blocks, threads>>>(
        temp, spectrum, M, N
    );

    cudaDeviceSynchronize();

    cudaFree(temp);
}


```