---
title: "Gaussian Blur: 高斯模糊"
description: "Gaussian Blur 题解"
date: 2026-10-13
tags:
    - Algorithm
    - CUDA
    - GPU
sidebar:
    order: 7
---

# Gaussian Blur: 高斯模糊
> **题目来源**：[LeetGPU — Gaussian Blur](https://leetgpu.com/challenges/gaussian-blur) · **难度**：Medium · **语言**：CUDA C++  

## 题目描述

使用 CUDA 实现一个二维图像的高斯模糊操作。

给定一个输入图像 `input` 和一个高斯卷积核 `kernel`，需要对图像中的每个像素应用高斯滤波，计算得到模糊后的输出图像 `output`。

对于每个输出像素，需要计算其邻域像素与高斯核对应位置元素的加权和。

---

## 输入

给定：

- `input`：二维输入图像，使用一维 `float` 数组存储
- `kernel`：二维高斯滤波核，使用一维 `float` 数组存储

输入图像尺寸：

```
input_rows × input_cols
```

卷积核尺寸：

```
kernel_rows × kernel_cols
```

数据均采用 **行优先（row-major）存储**。

输入图像索引：

```cpp
input[row * input_cols + col]
```

卷积核索引：

```cpp
kernel[row * kernel_cols + col]
```

---

## 计算方式

对于输入图像中的每一个像素：

```
(row, col)
```

计算：

```
output[row][col] =
    Σ input[row+i][col+j] * kernel[i][j]
```

其中：

- `(i, j)` 遍历整个卷积核
- 卷积核中心与当前像素对齐

例如，一个 `5×5` 的卷积核：

```
x x x x x
x x x x x
x x P x x
x x x x x
x x x x x
```

其中 `P` 表示当前计算的像素。

---

## 边界处理

当卷积核覆盖到图像边界之外时：

- 不允许访问非法内存
- 只计算图像范围内有效的像素

例如：

输入：

```
A B C
D E F
G H I
```

当卷积中心位于左上角：

```
X X X
X A B
X D E
```

其中：

```
X
```

表示超出图像范围的位置。

这些位置应该忽略，只计算：

```
A、B、D、E
```

---

## 函数接口

```cpp
extern "C" void solve(
    const float* input,
    const float* kernel,
    float* output,
    int input_rows,
    int input_cols,
    int kernel_rows,
    int kernel_cols
);
```

参数说明：

| 参数 | 描述 |
|------|------|
| `input` | GPU 上的输入图像数据 |
| `kernel` | GPU 上的高斯卷积核 |
| `output` | GPU 输出图像 |
| `input_rows` | 输入图像高度 |
| `input_cols` | 输入图像宽度 |
| `kernel_rows` | 卷积核高度 |
| `kernel_cols` | 卷积核宽度 |

---

## CUDA 实现要求

需要：

- 使用 CUDA kernel
- 利用 GPU 并行计算
- 每个线程计算一个输出像素

推荐线程映射：

```
threadIdx.x → column（列）

threadIdx.y → row（行）
```

计算方式：

```cpp
int row =
    blockIdx.y * blockDim.y
    + threadIdx.y;

int col =
    blockIdx.x * blockDim.x
    + threadIdx.x;
```

输出：

```cpp
output[row * input_cols + col]
```

---

## 约束条件

- 输入图像和卷积核类型均为 `float`
- 图像采用 row-major 存储
- 卷积核尺寸通常为奇数

例如：

```
3 × 3

5 × 5

7 × 7
```

---

## 示例

输入图像：

```
1 2 3
4 5 6
7 8 9
```

卷积核：

```
0 1 0
1 4 1
0 1 0
```

计算中心像素：

```
output[1][1]
```

计算：

```
=
1*0 + 2*1 + 3*0
+
4*1 + 5*4 + 6*1
+
7*0 + 8*1 + 9*0
```

---

## Naive

> Naive CUDA Kernel
> 建立二维kernel, 让每个线程计算自己的高斯模糊卷积

```cpp
#include <cuda_runtime.h>

__global__ void gaussian_blur(const float* input, const float* kernel, float* output, int input_rows, int input_cols, int kernel_rows, int kernel_cols) {
    int idx = blockDim.y * blockIdx.y + threadIdx.y;
    int idy = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= input_rows || idy >= input_cols) {
        return ;
    }

    float t = 0.0f;

    for (int m = -kernel_rows / 2; m <= kernel_rows / 2; m++) {
        for (int n = -kernel_cols / 2; n <= kernel_cols / 2; n++) {
            int newx = idx + m, newy = idy + n;
            if (newx >= 0 && newx < input_rows && newy >= 0 && newy < input_cols) {
                t += input[(idx + m) * input_cols + idy + n] * kernel[(m + kernel_rows / 2) * kernel_cols + n + kernel_cols / 2];
            }
        }
    }

    output[idx * input_cols + idy] = t;
}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output, int input_rows, int input_cols, int kernel_rows, int kernel_cols) {
    dim3 threads(16, 16);
    dim3 blocks((input_cols + 15) / 16, (input_rows + 15) / 16);
    gaussian_blur<<<blocks, threads>>>(input, kernel, output, input_rows, input_cols, kernel_rows, kernel_cols);
    cudaDeviceSynchronize();
}

```
---

## 优化: Shared Memory Tiling
Naive实现有一个重大的问题: 相邻的Thread会大量读取重复数据

```cpp

#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void gaussian_blur_shared(
    const float* input, const float* kernel, float* output,
    int input_rows, int input_cols, int kernel_rows, int kernel_cols
) {
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    int radius_y = kernel_rows / 2;
    int radius_x = kernel_cols / 2;

    extern __shared__ float tile[];

    int tile_rows = BLOCK_SIZE + 2 * radius_y;
    int tile_cols = BLOCK_SIZE + 2 * radius_x;

    for (int y = threadIdx.y; y < tile_rows; y += BLOCK_SIZE) {
        for (int x = threadIdx.x; x < tile_cols; x += BLOCK_SIZE) {
            int input_y = blockIdx.y * BLOCK_SIZE + y - radius_y;
            int input_x = blockIdx.x * BLOCK_SIZE + x - radius_x;

            float value = 0.0f;

            if (input_y >= 0 && input_y < input_rows &&
                input_x >= 0 && input_x < input_cols) {
                value = input[input_y * input_cols + input_x];
            }

            tile[y * tile_cols + x] = value;
        }
    }

    __syncthreads();

    if (row < input_rows && col < input_cols) {
        float sum = 0.0f;

        for (int m = -radius_y; m <= radius_y; m++) {
            for (int n = -radius_x; n <= radius_x; n++) {
                int ty = threadIdx.y + m + radius_y;
                int tx = threadIdx.x + n + radius_x;

                sum += tile[ty * tile_cols + tx] *
                       kernel[(m + radius_y) * kernel_cols +
                              (n + radius_x)];
            }
        }

        output[row * input_cols + col] = sum;
    }
}

extern "C" void solve(
    const float* input, const float* kernel, float* output,
    int input_rows, int input_cols, int kernel_rows, int kernel_cols
) {
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    dim3 blocks(
        (input_cols + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (input_rows + BLOCK_SIZE - 1) / BLOCK_SIZE
    );

    int radius_y = kernel_rows / 2;
    int radius_x = kernel_cols / 2;

    size_t shared_size =
        (BLOCK_SIZE + 2 * radius_y) *
        (BLOCK_SIZE + 2 * radius_x) *
        sizeof(float);

    gaussian_blur_shared<<<blocks, threads, shared_size>>>(
        input, kernel, output,
        input_rows, input_cols, kernel_rows, kernel_cols
    );

    cudaDeviceSynchronize();
}

```
