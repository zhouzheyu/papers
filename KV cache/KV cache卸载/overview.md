# overview

背景：长上下文的KV cache很快就超过了内存容量，迫使跨内存层次结构分层传输。

## Strata-osdi2026

* 高效I/O，打满PCIe带宽
  * 用GPU kenel（高并发）取代CPU发起的cudaMemcpyAsync API
  * 在GPU pool使用Layer-first（计算友好）的内存布局，在host memory使用Page-fist（传输友好）的内存布局
* cache感知的请求调度
  * 修改SGLang的前缀树，标记正在队列中的请求，避免重复计算
  * 根据请求所需的KV cache是否存在/是否在GPU中，在一个batch内混搭I/O bound和compute bound的请求
  * 即使采用了平衡批次，有些批次仍然可能是“loading bound”的。调度器的最终策略是在加载气泡中插入 Decoding Batch，用计算掩盖加载停顿

## SolidAttention-Fast2026

在个人笔记本上本地跑一个长上下文的 LLM，使用SSD缓存KV cache。

* KV Consolidator：粗粒度访问
  * Block-wise 注意力稀疏：Block 越大，SSD 读的粒度越大，带宽利用率越高，但精度下降
  * 改KV内存布局：把 K 和 V 交织在一起，传输单元翻倍，I/O 次数减半，SSD 带宽利用率 double。
* Speculative Prefetcher：提前加载KV cache
  * Key insight: 相邻层之间的 block 选择结果高度相似。
  * 用上一层的选择结果预测下一层需要哪些 block，提前发起 I/O。
  * 预取命中，提前加载到DRAM；预取失误，再从SSD里面读，并覆盖到错误块的位置。
* SSD-aware Scheduler：内存受限时，GPU 读数据和 SSD 预取数据可能发生冲突，怎么安全调度？
  * DAG 调度，同步点复用

## ECHO-osdi2026

* Graph-Friendly Cache Manager：把cache management塞进CUDA Graph，避免CPU launch，加速推理（特别是decode）
  * host是所有KV cache的全量备份，GPU pool存全部indexer以及选中token的KV cache。
  * 保住CUDA Graph的操作：
    * 元数据：因为CUDA Graph不允许动态创建或变长tensor，因此全部是静态预分配的定长tensor。
    * recall的时候用GPU线程并行读，而不是走 host 侧发起的拷贝。既避免了 CPU 干预，又提高了 PCIe 带宽利用率。
  * 因为host存的是全部KV cache，GPU驱逐KV cache的时候就直接不要了，不需要写回host memory。
* lossless KV prefetch
  * Intra-query prefetching（decode）：把 top-k 变成 top-p
  * Inter-query prefetching（prefill）：近似 top-k + 分数平移
  * prefetch融合kernel的三级流水线...
