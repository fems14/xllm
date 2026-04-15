Role & Context
你是一个资深的 AI Infra 工程师，精通大语言模型（LLM）推理加速、调度算法以及基于华为 Ascend NPU 的底层性能优化（CANN、ACL API）。
当前任务：将 vllm-ascend 框架中的 “Chunked-Prefill（分块预填）” 特性迁移到 xllm 框架中。

Objective
通过将长序列 Prefill 请求切分为多个 Chunk 执行，降低首 Token 延迟（TTFT）的抖动，优化长序列推理时的显存峰值压力，并提升 Prefill 与 Decode 混合调度时的系统吞吐。

Workflow Constraint
请严格按照以下 4 个阶段执行，每个阶段结束后必须等待我的 review 和确认（输入 "APPROVE"），才能进入下一阶段。

Phase 1: Feature Analysis (vllm-ascend)
请分析 vllm-ascend 中 Chunked-Prefill 的实现机制（重点关注 Scheduler 的切分逻辑、AttentionMetadata 的构建以及 vLLM 如何处理跨 Chunk 的 KV Cache 读写）。

总结该特性的核心技术点：

切分策略： 如何确定 chunk_size？如何处理最后一个不满足 chunk_size 的残余序列？

算子适配： 算子（如 FlashAttention）如何接收 q_seq_lens（当前 chunk 长度）与 kv_seq_lens（已累积长度）不一致的情况？

Metadata 转换： 在 C++ 侧如何构建 q_cu_seq_lens 和 kv_cu_seq_lens 以适配变长算子？

输出一份简短的架构分析报告，特别指出针对昇腾 NPU 的特殊优化（如内存对齐要求）。

Phase 2: Design Document (xllm)
基于 Phase 1 的分析结果以及 xllm 的现有架构，输出一份技术设计方案：

架构适配： xllm 的 AttentionMetadataBuilder 如何扩展以支持多 Chunk 状态维护？

算子调用栈： 描述在 xllm 中调用底层 NPU 算子（如 npu_fusion_attention）时，参数 q_seq_lens、kv_seq_lens 以及 paged_attention 的传参变化。

AclGraph 兼容性： 重点设计如何在 Graph 捕获模式 下构建动态长度的 Metadata。必须严格避免在算子执行路径中使用 .item() 或 aclrtMemcpy 等同步操作。

KV Cache 管理： 描述 Chunked-Prefill 过程中，中间状态的 KV Cache 如何在 xllm 的 BlockManager 中进行增量填充。

等待我确认该设计。

Phase 3: Implementation
按照确定的 Phase 2 设计，分步骤修改 xllm 代码：

修改 xllm 的调度层逻辑，支持将一个物理 Request 拆分为多个待执行的推理 Task。

在 C++ 侧完善 AttentionMetadata 的构建逻辑（重点实现基于 torch::cumsum 和 torch::cat 的无同步前缀和计算）。

适配模型层的 Attention 转发逻辑，确保支持 q_len < kv_len 的增量 Prefill 模式。

处理好 Prefill 结束后的 slot_mapping 更新，确保后续 Decode 阶段能正确寻址。

Phase 4: Testing & Verification
编写或修改针对 xllm 的 Chunked-Prefill 单元测试，覆盖不同 chunk_size（如 128, 256, 512）。

重点验证：

(A) 精度一致性： 验证开启 Chunked-Prefill 后，输出的 Logits 与标准 Prefill 完全对齐。

(B) KV Cache 正确性： 确认各 Chunk 写入的 KV 值在显存中连续且未被覆盖。

(C) 性能验证： 使用 Ascend Insight 或 profiling 工具确认没有因 Metadata 构建导致的 CPU/NPU 同步空隙（Gap），观察算子执行流是否连贯。

现在，请开始执行 Phase 1。如果在分析过程中你需要我提供特定的文件路径或入口函数，请向我提问。
