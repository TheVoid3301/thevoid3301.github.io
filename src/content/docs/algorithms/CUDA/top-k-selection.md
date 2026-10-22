---
title: "Top-K Selection：分块排序与分层归并"
description: "Top-K Selection"
date: 2026-10-22
tags:
    - Algorithm
    - CUDA
    - GPU
sidebar:
    order: 8
---

# Top-K Selection：分块排序与分层归并

> **题目来源**：[Top-K Selection](https://leetgpu.com/challenges/top-k-selection) · **难度**：Medium · **语言**：CUDA C++  

## 题目描述

给定一个包含 (N) 个单精度浮点数的一维数组 input，以及一个整数 (k)，要求在 GPU 上选出数组中最大的 (k) 个元素，并将结果按照降序排列，写入 output。

输入

input：GPU 上的浮点数数组。

N：输入数组的长度。

k：需要选取的最大元素数量。

输出

output：包含最大的 (k) 个元素，按照从大到小的顺序排列。

约束与接口

(1 \le N \le 100,000,000)

(1 \le k \le N)

数组元素使用 32 位浮点数。

实现需要考虑大规模输入的性能。

函数接口：

extern "C" void solve(
    const float* input,
    float* output,
    int N,
    int k
);

示例

输入：

input = [3.0, 8.0, 1.0, 7.0, 5.0, 2.0]
N = 6
k = 3

输出：

output = [8.0, 7.0, 5.0]

## Naive

> Naive CUDA Kernel

```cpp
#include <cuda_runtime.h>

// 将最小堆向下调整
__device__ void sift_down(float* heap, int p, int size) {
    while (true) {
        int left = 2 * p + 1;
        int right = left + 1;
        int smallest = p;

        if (left < size && heap[left] < heap[smallest])
            smallest = left;

        if (right < size && heap[right] < heap[smallest])
            smallest = right;

        if (smallest == p)
            break;

        float t = heap[p];
        heap[p] = heap[smallest];
        heap[smallest] = t;

        p = smallest;
    }
}

__global__ void topk(
    const float* input,
    float* output,
    int N,
    int k
) {
    // 简单版本：仅使用一个 GPU 线程
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;

    int size = 0;

    for (int i = 0; i < N; ++i) {
        float x = input[i];

        // 堆未满，直接插入并向上调整
        if (size < k) {
            int p = size++;
            output[p] = x;

            while (p > 0) {
                int parent = (p - 1) / 2;

                if (output[parent] <= output[p])
                    break;

                float t = output[parent];
                output[parent] = output[p];
                output[p] = t;

                p = parent;
            }
        }
        // 堆已满，只有比堆顶更大的元素才需要处理
        else if (x > output[0]) {
            output[0] = x;
            sift_down(output, 0, size);
        }
    }

    // 最小堆排序，最终得到降序结果
    for (int end = k - 1; end > 0; --end) {
        float t = output[0];
        output[0] = output[end];
        output[end] = t;

        sift_down(output, 0, end);
    }
}

extern "C" void solve(
    const float* input,
    float* output,
    int N,
    int k
) {
    if (N <= 0 || k <= 0 || k > N)
        return;

    topk<<<1, 1>>>(input, output, N, k);
}

```