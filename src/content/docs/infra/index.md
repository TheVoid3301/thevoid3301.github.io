
---
title: AI Infrastructure
description: GPU 编程、大模型推理、系统底层与高性能计算
sidebar:
  order: 1
---

# ⚡ AI Infrastructure

> Understand the hardware. Master the systems. Push the limits.

探索人工智能基础设施的底层实现，从 GPU 编程到大模型推理引擎，从算子优化到分布式计算。

这里记录我在 **AI Infra、系统编程与高性能计算** 方面的学习与实践。

---

## 🖥️ GPU Programming

深入 GPU 架构与并行计算模型。

### CUDA

- CUDA Programming Model
- Thread / Block / Grid
- Global Memory / Shared Memory / Registers
- Memory Coalescing
- Warp Shuffle & Reduction
- Bank Conflict
- Tensor Core & WMMA
- CUDA Kernel Optimization

### Triton

- Triton Programming Model
- Blocked Layout
- Vectorization
- Matrix Multiplication
- Softmax & LayerNorm
- Attention Kernel
- Kernel Fusion
- Performance Benchmarking

## 🧠 Deep Learning Systems

理解深度学习框架的核心机制。

- PyTorch Autograd
- Tensor & Memory Management
- Computational Graph
- Torch Compile
- Distributed Training
- Mixed Precision
- Operator Implementation

## 🚀 LLM Inference

研究大语言模型的推理架构与性能优化。

- Transformer Architecture
- Multi-Head Attention
- RoPE & Position Encoding
- KV Cache
- PagedAttention
- FlashAttention
- Continuous Batching
- Tensor Parallelism
- Speculative Decoding

## ⚙️ Systems & Performance

从系统底层理解性能。

- C++ / Rust
- Operating Systems
- Computer Architecture
- Memory Hierarchy
- SIMD / SIMT
- Compiler Optimization
- Profiling & Benchmarking

---

## 🎯 Learning Philosophy

**Understand → Implement → Benchmark → Optimize**

不仅关注算法如何工作，更关注：

1. 为什么这样设计？
2. 底层硬件如何执行？
3. 性能瓶颈在哪里？
4. 是否存在更高效的实现？

> Performance is not magic. It is engineering.
