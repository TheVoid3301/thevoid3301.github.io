---
title: "Norm 与 LU"
description: "Norm 与 LU"
date: 2026-10-10
tags:
    - Project
    - VLLM
    - LLM
    - GPU
    - Norm
    - LU
sidebar:
    order: 2
---

# Norm 与 LU
> 对比 RMSNorm 与 LayerNorm 的公式差异及工程取舍。
> 理解 rsqrt、混合精度路径（先 float 再回写 dtype）的原因。
> 掌握 add_rms_forward：残差与归一化的融合顺序及为何能省一次大张量读写。
> 理解 SwiGLU：SiluAndMul 与 gate_up_proj 如何拼成门控前馈。

## LayerNorm

假设Token的Hidden State 为
$$
x=[x_1,x_2,\dots,x_d]
$$

LayerNorm分为以下三步

### 1.计算均值
$$
\mu=\frac{1}{d}\sum_{i=1}^{d}x_i
$$
### 2.计算方差
$$
\sigma^2=\frac{1}{d}\sum_{i=1}^{d}(x_i-\mu)^2
$$
### 3.归一化, 之后进行可学习的仿射变换
$$
\mathrm{LN}(x) = \gamma  \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$
其中：
- \(\gamma\)：可学习的缩放参数
- \(\beta\)：可学习的偏移参数
- \(\epsilon\)：防止除零的极小值
- \(d\)：Hidden State 的维度

---

## RMSNorm
RMSNorm对比LayerNorm最大的不同就是, 它不需要减去均值, 也就不需要计算均值.

### 1.计算平方的均值
$$
m_2=\frac{1}{d}\sum_{i=1}^{d}x_i^2
$$

### 2.开方得到RMS
$$
\operatorname{RMS}(x)=\sqrt{m_2+\epsilon}
$$

### 3.归一化并加上可学习参数
$$
\mathrm{RMSNorm}(x) = \gamma \frac{x}{\mathrm{RMS}(x)}
$$

标准RMSNorm中没有\(\beta\)偏移参数

---

## 为什么用 rsqrt 而不是 1/sqrt
### 1.GPU有专门支持倒数平方根的硬件指令
### 2.尽可能减少中间计算, 有利于融合和优化
### 3.归一化可以复用同一个倒数平方根结果, 也就是之后可以转化成大量并行的乘法

---

## 为什么RMSNorm要先转化成FP32
### FP32有效精度高

---

## 为什么要转换回dtype
### 1.保持模型各层的数据结构一致
### 2.适当低精度运算可以降低数据传输量, 提高吞吐能力

---

## add_rms_forward：为什么融合 Residual Add 和 RMSNorm
当存在 `residual` 时：

1. `x = x.float().add_(residual.float())`：在**高精度**上做 **\(x \leftarrow x + \text{residual}\)**（此处命名上 `x` 是子层输入、`residual` 是上一分支未归一化的流）
2. `residual = x.to(orig_dtype)`：把**相加后的主路径**保存为下一轮子层的「残差分支」（典型 Pre-Norm：**norm 只对一支做，残差另一支绕开**）
3. 对相加后的 `x` 做 RMSNorm（同 `rms_forward` 的后半段）

这与「先 norm 再子层再加残差」在数学上需与整体 `forward` 编排一致；本仓库通过 `forward` 分流 `residual is None` 实现第一层与后续层的差异（见下一节源码）
