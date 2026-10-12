---
title: "KV Cache 原理与实现"
description: "QKV Cache 原理与实现"
date: 2026-10-12
tags:
    - Project
    - VLLM
    - LLM
    - GPU
    - Qwen3
    - KV Cache
sidebar:
    order: 5
---

# KV Cache 原理与实现

> 自回归解码每一步只新增一个 token，却若每次都从头重算整段序列的 K/V，会浪费 \(O(T^2)\) 级别的重复算力；KV Cache 把历史位置的 K、V 存下来，让解码步近似 \(O(T)\) 地增长.

## KV Cache 的概念

### 1. 注意力里重复计算从何而来

对长度 \(T\) 的序列，第 \(t\) 步若从零计算 attention，需要所有位置 \(1..t\) 的 K、V 参与。但 **第 \(t\) 步新增的只是位置 \(t\) 的 query**；位置 \(1..t-1\) 的 K、V 与上一步相比 **不变**（模型权重与已生成 token 固定时）。

因此可把 **过去所有步已算过的 K、V** 缓存在 GPU 上，本步只算当前 token 的 K、V 并 **追加** 到 cache，再与历史一起做注意力（常配合因果掩码或「只 attend 到过去」的实现）。

**省的是什么**：避免对历史 token 重复做 K/V 投影与（在部分实现中）重复写入中间结果。复杂度从「每步像做一次长序列 prefill」降为「每步常数级或线性于当前上下文」的增量更新（具体常数依赖 kernel 与 head 配置）。

### 2. 经典显存估算公式（与课程一致）

对每层、每个序列位置，需要存 **K** 与 **V** 两个张量，形状与 `num_kv_heads`、`head_dim` 相关.

粗算 **总 KV Cache 字节数**（与下面 nano-vllm 实现维度一致时可写作）：

$$
\text{KV\_bytes} \approx 2 \times L \times T \times H_{kv} \times D \times S_{\mathrm{dtype}} \times B
$$

其中：

- **2**：K 与 V 两份；
- **\(L\)**：`num_hidden_layers`；
- **\(T\)**：序列长度（或当前占用的 token 数上限）；
- **\(H_{kv}\)**：`num_key_value_heads`（注意 **张量并行后要按 rank 取本地头数**）；
- **\(D\)**：`head_dim`；
- **\(S_{\mathrm{dtype}}\)**：单元素字节（fp16=2，bf16=2 等）；
- **\(B\)**：batch 或并行序列数（依系统是否共享/分槽而定）.

### 3. nano-vllm 中的「块」与全局张量

nano-vllm 不是为「每个请求 malloc 一块连续 KV」的简单模式，而是预先分配 **固定块数 × 块大小** 的大池，由 **BlockManager** 管理映射（下一课）。`allocate_kv_cache` 负责 **池子本体** 的显存与 **按层绑定** 到各 attention 模块的 `k_cache` / `v_cache` 视图.

### 4. Prefill vs Decode

- **Prefill（提示阶段）**：一次性处理 prompt，可并行算多个 token 的 Q/K/V，向 KV Cache **写入** 一段连续区间；计算形态常为「大张量、高并行」.
- **Decode（生成阶段）**：每步通常只处理 **1 个新 token**（或少量），对 KV Cache **增量追加**，计算形态常为「小 batch、内存带宽敏感」.

KV Cache 在 decode 阶段的收益最大：若无 cache，每步都要对全长重算历史 K/V，延迟爆炸.

## 源码解析

> nano-vllm/nanovllm/engine/model_runner.py

```python
    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1
```