Role & Context你是一个资深的 AI Infra 工程师，负责 xllm 框架在华为 Ascend NPU 上的性能优化与架构对齐。当前背景： xllm 已支持通用模型的 Chunked-Prefill 和 Qwen3.5 基础架构。当前任务： 实现 Qwen3.5 Gated DeltaNet (GDN) 层的分块预填适配。核心是通过复用 xllm 现有的 Token 计数器参数（如 num_computed_tokens 或 history_lens），驱动 GDN 算子在分块场景下的正确分支切换。
Objective参考 vllm-ascend 的实现，利用 xllm 已有的分块调度参数，使 Qwen3.5 的 GDN 算子能自动识别并衔接跨分块的 Recurrent State。确保计算过程符合 AclGraph 零同步要求，且精度与标准 Prefill 完全对齐。
Workflow Constraint请严格按照以下 4 个阶段执行，每个阶段结束后必须等待我的 review 和确认（输入 "APPROVE"），才能进入下一阶段。
Phase 1: Infrastructure & Kernel Discovery计数器参数探测： 调研 xllm 中已支持 Chunked-Prefill 的模型（如 Llama），找出用于追踪“已计算 Token 数”的核心参数名及传递链路（重点查看 AttentionMetadata 结构体）。vLLM 算子分流参考： 分析 vllm-ascend 中 Qwen3.5 GDN 算子的分支逻辑：分支 A (Init)： 当已计算 Token 数为 0 时，调用算子进行全量计算并初始化 State。分支 B (Update)： 当已计算 Token 数 > 0 时，从 Cache 中读取 last_recurrent_state 作为 initial_state 传入算子，执行增量更新逻辑。算子能力确认： 检查 xllm 当前封装的 GDN 算子接口，是否支持 initial_state 传参及 q_len > 1 时的增量计算模式。输出报告： 说明参数映射关系及算子调用策略，提议如何无缝“挂载”到现有体系。
Phase 2: Design Document (Stateful Routing & Zero-Sync)基于 Phase 1 的调研结果，设计 Qwen3.5 的技术适配方案：状态寻址与路由：设计在处理非首个 Chunk 时，如何利用 slot_mapping 和计数器偏移量从 GDN Cache 中精准提取前序状态。明确算子在不同 Chunk 阶段的分流入口（Init vs Update）。AclGraph 零同步设计：核心挑战： 必须规避导致捕获模式崩溃的 .item() 或同步 aclrtMemcpy。方案： 在 C++ 侧（AttentionMetadataBuilder）使用纯 Tensor 算子（如 torch::cumsum、torch::cat、torch::where）构建分支判断掩码和偏移量 Tensor。显存复用： 确认分块过程中的中间状态是否直接复用现有的 Decode GDN Cache 空间，确保显存管理的连贯性。等待我确认该设计。
Phase 3: Implementation (Infrastructure Integration)按照 Phase 2 的设计，分步骤修改 xllm 代码：C++ Metadata Builder： 在 attention_metadata_builder.cpp 中实现对计数器参数的处理，生成支持分块的 q_cu_seq_lens 和状态索引，严禁引入任何同步点。Model Layer 逻辑挂载： 修改 Qwen3.5 的 Layer 转发逻辑。引入发现的计数器参数作为判断条件，将后续 Chunk 导向 GDN 的增量处理分支。算子调用对齐： 适配 GatedDeltaNet 算子接口，确保在 q_len > 1 且有历史状态时，正确执行状态加载与写回。状态流转闭环： 确保最后一个 Chunk 产出的状态能被 Decode 阶段无缝识别并作为起始输入。
Phase 4: Testing & Verification分块精度回归： 开启 Chunked-Prefill，对比长序列（如 2k+ tokens）在分块执行与一次性执行下的输出误差（Logits 误差需 $< 10^{-5}$）。零同步压力测试： 在 NPU 环境下开启 Graph 捕获模式运行，确认预热和执行阶段无任何运行时同步报错。Profiling 性能分析：使用 Ascend Insight 确认后续分块确实加载了前序状态。检查算子执行流是否连贯，确认没有因逻辑分流引入额外的 CPU 下发延迟（Gap）。
现在，请开始执行 Phase 1。请先去调研 xllm 中现有 Chunked-Prefill 的计数器参数及其在 Qwen3.5 算子上的挂载可行性。



# Role & Context
你是一个资深的 AI Infra 工程师，精通大语言模型（LLM）推理加速、MoE 架构以及基于华为 Ascend NPU 的底层性能优化（CANN、ACL API）。
当前任务：将 `vllm-ascend` 框架中的 “MoE 共享专家多流（Shared Expert Multi-stream）” 特性迁移到 `xllm` 框架中。

# Objective
通过重叠 Shared Expert 和 Routed Experts 的计算，隐藏 NPU 算子执行 latency，提升整体吞吐。

# Workflow Constraint
请严格按照以下 4 个阶段执行，每个阶段结束后必须等待我的 review 和确认（输入 "APPROVE"），才能进入下一阶段。

## Phase 1: Feature Analysis (vllm-ascend)
1. 请分析本工作区中 `vllm-ascend` 的相关代码（重点关注 MoE execution, stream assignment, ACL event synchronization 相关的逻辑）。
2. 总结该特性的核心技术点：流的创建与生命周期、算子下发到哪个流、同步点（aclrtRecordEvent/aclrtStreamWaitEvent）设置在何处。
3. 输出一份简短的架构分析报告。

## Phase 2: Design Document (xllm)
基于 Phase 1 的分析结果以及 `xllm` 的现有架构，输出一份技术设计方案。设计方案必须包含：
1. **架构适配：** `xllm` 中现有的算子派发机制（Dispatch）如何支持多流？是否需要扩展底层的 Stream 管理器？
2. **时序图：** 描述 Shared Expert 计算与 Routed Experts 计算在多流下的并行生命周期和同步点。
3. **显存与资源约束：** 多流并发执行时，临时 Workspace 显存是否有冲突风险？如果有，如何解决？
4. 等待我确认该设计。

## Phase 3: Implementation
按照确定的 Phase 2 设计，分步骤修改 `xllm` 代码：
1. 先实现或完善底层的 NPU Stream/Event 管理接口（如果 `xllm` 缺失）。
2. 在 MoE 相关的模型层（如 Qwen/DeepSeek 等包含 MoE 的模型结构中）接入多流计算逻辑。
3. 处理好 forward pass 结束时的流同步（Stream Synchronize），确保后续层拿到的 tensor 数据一致。

## Phase 4: Testing & Verification
1. 编写或修改针对 `xllm` 的 MoE 单元测试。
2. 重点验证：(A) 结果精度对齐（与单流执行的 logits 误差为0）；(B) 确认算子下发没有死锁（Deadlock）；(C) 通过 profiling 工具检查并发效果。

现在，请开始执行 Phase 1。如果在分析过程中你需要我提供特定的文件路径或入口函数，请向我提问。
