
---
title: Projects
description: 工程项目、源码分析与系统实现
sidebar:
  order: 1
---

# 🛠️ Projects

> Build things. Break things. Understand everything.

这里记录我的工程项目、开源项目学习和系统实现过程。

相比单纯使用工具，我更希望理解其内部原理，并尝试从零构建核心模块。

---

## 🚀 LLM Systems

### nano-vLLM

轻量级大语言模型推理引擎。

**探索方向：**

- Model Loading
- Tokenization
- Model Execution
- KV Cache Management
- Continuous Batching
- Tensor Parallelism
- CUDA Graph
- Sampling Optimization

**技术栈：** Python · PyTorch · CUDA · Triton

### Inference Engine Exploration

研究高性能 LLM Serving 系统。

- Request Scheduling
- Memory Management
- Prefill / Decode
- Model Parallelism
- Throughput & Latency
- Serving Architecture

---

## ⚡ GPU Kernel Engineering

从零实现并优化 GPU 算子。

### CUDA Kernels

- Vector Addition
- Parallel Reduction
- Softmax
- Matrix Multiplication
- LayerNorm
- Attention

### Triton Kernels

- Fused Operations
- Tiled Matrix Multiplication
- FlashAttention
- Memory Optimization
- Kernel Benchmarking

---

## 🖥️ Systems Programming

探索操作系统、编译器和高性能系统软件。

- C++ Systems Programming
- Rust Systems Programming
- Memory Allocators
- Concurrency & Parallelism
- Compiler Internals
- Storage & Database Systems

---

## 🔬 Experimental Projects

记录原型系统、实验性架构和探索性实现。

每个项目尽可能包含：

1. **Motivation** — 为什么要做？
2. **Architecture** — 系统如何设计？
3. **Implementation** — 核心实现过程
4. **Challenges** — 遇到了哪些问题？
5. **Benchmark** — 如何测量性能？
6. **Lessons Learned** — 获得了什么经验？

---

## 💡 Engineering Philosophy

**Read Code → Rebuild → Measure → Improve**

代码不仅要能够运行，也应当尽可能做到：

- Correctness
- Readability
- Performance
- Maintainability

> Talk is cheap. Show me the code.
