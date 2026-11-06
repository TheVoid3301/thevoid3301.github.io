---
title: "现代语言模型架构演进"
description: "现代语言模型架构演进"
date: 2026-11-6
tags:
    - LLM
    - Transformer
    - Modern LLM
sidebar:
    order: 1
---

# 现代语言模型架构演进
> 此文档为 [MODERN LLM NOTEBOOK](https://walkinglabs.github.io/modern-llm-notebook/?utm_source=chatgpt.com#08-gpt2-to-modern-models) 的学习笔记

---

## 深层网络的归一化和残差链接
### Post-Norm
```python
x = x + attention(x)
x = norm(x)
```
LayerNorm在Attention之后, 所以叫Post-Norm

计算图演示:
```python
  x ──→ Attention ──→ + ──→ LayerNorm ──→ 输出
  │                   ↑
  └───────────────────┘  (残差连接)
```
残差连接加到 Attention 输出上，然后两者一起经过 LayerNorm. 也就是说，残差路径经过了 LayerNorm.

残差连接的作用是让梯度有一条直接回传的路径，不被 Attention 和 FFN 衰减.

但在 Post-Norm 里，这条路径上插入了 LayerNorm. LayerNorm 会缩放梯度，经过它之后梯度大小可能被改变.

在 4 层网络里这不是问题.但在 40 层、80 层网络里，每层都经过一次 LayerNorm，梯度的缩放效应会累积，可能爆炸或消失.

---

### Pre-Norm
```python
x = x + attention(norm(x))
```
```python
  x ──→ LayerNorm ──→ Attention ──→ + ──→ 输出
  │                                 ↑
  └─────────────────────────────────┘  (残差连接，不经过 Norm)
```
关键变化是残差连接绕过了 LayerNorm，梯度在残差路径上不被缩放. 能够在一定程度上有效防止多层网络中容易出现的梯度爆炸或者梯度消失.

---

### RMSNorm
RMSNorm 的核心公式：
```python
RMS(x) = √( (x₁² + x₂² + x₃² + x₄²) / 4 )

RMSNorm(x) = x / RMS(x) × γ
```
对比LayerNorm, 可以不用算偏差平方, 也去掉了编译参数β

---

### SandwichNorm
```python
旧（标准 Pre-Norm）:
  x → RMSNorm → Attention → + 残差 → RMSNorm → FFN → + 残差
  (2 个 Norm)

新（Sandwich Norm, Gemma 3）:
  x → RMSNorm → Attention → RMSNorm → + 残差 → RMSNorm → FFN → RMSNorm → + 残差
  (4 个 Norm，子层输出也归一化)
```

---

### Hyper-Connections 与 mHC
Hyper-Connections（HC, ByteDance, 2024）把单条残差流扩宽为 n 条并行的流。深度方向的连接决定每条流从上一层取多少（depth-connection），宽度方向的连接让 n 条流在同一层内互相交换（width-connection）.两组权重都可学习，相当于把「残差怎么接线」交给模型自己决定.

---

## Feed-Forward Network 与门控机制
### ReLU
```python
FFN(x) = W₂ · ReLU(W₁ · x)

ReLU 做的事：负数 → 0，正数 → 原样通过
```

### SwiGLU 门控机制
ReLU 把负的预激活值直接变成 0，这些位置在当前输入附近的梯度也为 0.

门控的思路是用两条并行通道：

一条做线性变换得到「信息」
另一条做线性变换 + Swish，得到一组平滑的门控值
两者按元素相乘，让权重决定每个分量放多少
效果是：如果某个分量部分有用，门控给出 0.3 的权重，它还是能通过一部分，而不是完全消失.

```python
教学版 FFN:  x → W1 → ReLU → W2 → 输出

SwiGLU FFN:  x → W_up   ┐
             x → W_gate ├→ 相乘 → W_down → 输出

SwiGLU(x) = (W_up · x)  ⊙  Swish(W_gate · x)
              ↑ 信息通道       ↑ 门控通道
              
Swish(a) = a × sigmoid(a) = a / (1 + e^(-a))
```

---

### Dense FFN 与 MoE