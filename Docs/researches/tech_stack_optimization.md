# ZenJev 技术栈优化与 Rust 化重构方案

**状态：** 架构决策记录 + 已落地部分实现
**日期：** 2026-09-21
**权威蓝图：** `Docs/stage0_zenjev_blueprint.md`（本文件是研究/设计记录，不改变 checklist 状态）

## 1. 结论先行

1. **模型平面保持 Python**：GLiNER2、PyTorch、PEFT、Transformers、CUDA kernel、tokenizer 与蒸馏逻辑用 Rust/Go 重写没有收益、只有风险（生态不可替代、数值兼容不可保证、迭代速度骤降）。
2. **服务平面全部 Rust 化**：MQ 协议桥、无限回放、背压/ACK/WAL、状态心跳、进程监督与崩溃重启、指标采集、监控工具、健康检查——这些是稳定性瓶颈所在，且已有 Rust 基础（`mq/`），继续加深。
3. **Go 暂不引入**：监控/桥接用 Rust 可与 `mq/` 共享工具链、无 GC 抖动、单静态二进制；仅当未来需要 Web 控制台/多租户 API 时再评估 Go（矩阵见 §7）。当前机器没有 Go 工具链，引入第三个工具链会直接损害"易于维护"。
4. **边界稳定比语言更重要**：把 Python 模型进程当作"可崩溃、可重启、无状态化到检查点"的 worker，Rust 作为 supervisor + 协议网关，崩溃不会传染，升级可灰度。

## 2. 现状盘点（按可迁移性）

| 层 | 现有实现 | 迁移判断 | 理由 |
|---|---|---|---|
| MQ 协议/回放/ACK/WAL/背压 | `mq/`（Rust，已实现） | ✅ 已完成 | 纯 IO/协议，Rust 强项 |
| 监控/指标采样 | `monitor/`（Rust，本次新增） | ✅ 已完成 | 高频采样、低开销、必须不崩溃 |
| 状态心跳 | `jev/status.py` + runtime/training 集成（本次新增） | ✅ 边界已定义 | JSON 契约，跨语言只读 |
| 训练/推理执行 | `jev/training.py`、`jev/runtime.py`、`jev/model.py` | ❌ 保留 Python | 依赖 PyTorch/GLiNER2/PEFT |
| Schema/蒸馏/校验 | `jev/config.py`、`jev/distill.py`、`jev/tool_task.py` | ⚠️ 部分可迁移 | 纯数据校验可复制到 Rust，但会形成双实现；优先保持单一真源 |
| CLI/运维面 | `jev/cli.py` | 🔜 未来用 Rust 网关包裹 | 调用模型进程，本身无状态 |
| 数据源连接器 | `jev/sources.py` | ⚠️ 可迁移 | HTTP/JSONL/RSS 纯 IO，但量小、非瓶颈 |
| 控制平面/监督 | 无（长时间靠单进程） | 🔜 高优先级 | 崩溃恢复、看门狗、重启退避 |

## 3. 目标分层

```
                       ┌─────────────────────────────────────────────┐
   operator ─────────► │ zenjev 网关（Rust，规划）                    │
                       │ 健康检查 / 状态聚合 / 启停 / 灰度重启          │
                       └──────────────┬──────────────────────────────┘
                                      │ 监督(Supervisor) + 心跳
        ┌─────────────────────────────┼──────────────────────────────┐
        ▼                             ▼                              ▼
┌───────────────┐            ┌─────────────────┐            ┌──────────────────┐
│ mq bridge     │  UDS credit│ model worker    │  JSONL/PT  │ zenjev-monitor   │
│ Rust          │───────────►│ Python + CUDA   │◄───────────│ Rust             │
│ 协议/WAL/ACK  │  + ACK     │ 训练/EMA/推理    │ status.json│ CPU/GPU/显存/代数 │
│ 无限 loop     │            │ GLiNER2/PEFT    │            │ JSONL 证据        │
└───────────────┘            └─────────────────┘            └──────────────────┘
```

关键点：**Python 只负责"算"，Rust 负责"跑不坏"**。所有跨边界交互都是版本化 JSON + Unix socket / 原子文件，Python 崩溃不影响桥与监控，桥只会在 ACK 前重投。

## 4. 本次已落地（可直接用）

1. **MQ 无限测试模式**（`mq/src/source/mock.rs`、`jev/mq.py`、`jev/cli.py`）
   - `loop: true` 时数据首尾相连；每个 cycle 注入 `loop_cycle` 并重算 `content_sha256`，因此幂等键不重复、去重不会误杀回放。
   - `max_cycles: 0` 无限；`max_cycles: N` 有界。状态文件持久化 cycle/cursor，重启续跑。
   - 入口：`python -m jev.cli mq run --loop --duration 60`、`scripts/smoke_mq_loop.py`。
2. **常驻型一体化 soak**（`scripts/run_infinite_soak.py`）
   - 无限 MQ → LoRA 训练 → EMA 发布 → 并发推理，时间有界但可去掉 `--duration` 变成永久模式。
   - 45 秒实测：200 条消费、16 个 loop cycle、200 训练步、推理 22 次、服务代数 1→133、0 错误。
3. **状态心跳契约**（`jev/status.py`，原子写 `runs/jev/status.json`）
   - 字段：`phase/generation/ema_step/training_steps/reset_id/loss/checkpoint/config_digest/model_id/device/pid`。
4. **Rust 监控工具**（`monitor/`，`zenjev-monitor`）
   - CPU（/proc/stat 差分）、内存、load、进程 RSS/CPU、GPU 利用率/显存/温度/功耗（nvidia-smi，缺 GPU 自动降级）、训练代数/EMA/loss/状态新鲜度。
   - `--format json`、`--jsonl`、`--once`、`--duration`；任何采样失败都跳过，不 panic。
5. **健壮性修复（重要）**：`ContinuousLoRAStream.submit` 在训练写线程死亡时会永久阻塞 → 现在快速失败并携带根因（`jev/training.py`），soak 由"挂死"变为显式报错。
6. **训练样本契约补全**：`training_example_from_envelope()` 统一 `labelled_example` 的 `example` / `labels+text` 两种形态，避免把无输入标签喂进训练（本次 soak 暴露的真实 bug）。

## 5. 建议的迁移路线（按收益/风险排序）

| 阶段 | 内容 | 收益 | 风险 | 验收 |
|---|---|---|---|---|
| P1（已做） | loop 模式、状态心跳、monitor、liveness 修复 | 可观测、可长跑、崩溃可发现 | 低 | `smoke_mq_loop`、soak、monitor 测试 |
| P2 | Rust `zenjev-supervisor`：启动/守护 Python worker，崩溃指数退避重启，读取 status 判活，超时杀进程组，重启后从 `latest.pt` 续训 | 服务不中断；Python 崩溃自动恢复 | 中：需要明确幂等边界 | 杀 worker 后自动恢复、续训 steps 不回退、证据 JSON |
| P3 | Rust 网关：把 `jev.cli` 的运维面（validate/manifest/mq/lag/dlq/replay）做成单二进制，内部 via UDS 调 worker；HTTP 健康/指标端点 | 运维面不依赖 Python 启动；部署单文件 | 中：双入口需去重 | 与 Python CLI 输出对拍测试 |
| P4 | 数据源连接器、schema digest 校验等纯 IO/校验下沉 Rust，Python 只保留模型路径 | 减少 Python 依赖面 | 中：双实现漂移 | 契约对拍 + 差分测试 |
| P5（可选） | Go 编写 Web 控制台/多租户 API（仅当有真实需求） | 生态成熟、上手快 | 引入第三工具链 | 独立部署、不影响主链路 |

**不建议**：用 Rust 重写 `gliner2_step`/EMA/反向传播（PyTorch 生态、数值等价与迭代速度无法接受）；用 Go 重写桥接与监督（与既有 Rust 重复造轮子）。

## 6. 健壮性与可维护性清单（设计约束）

健壮性：

- **崩溃隔离**：模型 worker 可被杀；Rust 侧只重投未 ACK 记录，绝不丢数据（at-least-once + 幂等键）。
- **有界资源**：credit 窗口、WAL 上限、队列上限、队列满时快速失败而非死锁。
- **原子状态**：所有持久状态（checkpoint/status/receipt/dedup）temp+fsync+rename。
- **失败即显式**：禁止"静默继续"；不合法 envelope 进 DLQ/quarantine，训练样本不合法进 quarantine。
- **降级可观测**：无 GPU/nvidia-smi 时监控仍工作并标注 `gpu: null`；状态过期标 `stale`。
- **可恢复性演练**：collapse/reset、crash-before-ack、loop 回放、重启续跑都要有测试与证据。

可维护性：

- **单一契约真源**：`JevConfig`/`MqConfig` 的 `as_contract()` 与 envelope/状态 JSON 是唯一协议文档；跨语言实现必须对拍。
- **二进制工具化**：每个关注点一个二进制（bridge/monitor/supervisor），参数手写解析、无框架、无隐藏依赖。
- **文档贴代码**：每个 crate 有 README + `Docs/runbooks/*` 操作手册；协议字段在 runbook 里逐字段解释。
- **测试分层**：Rust 单元/集成（协议/回放/解析）、Python pytest（契约/训练）、GPU smoke（真实模型）、soak（长跑）。
- **版本化协议**：`protocol`/`schema_version`/`monitor_schema_version` 已存在，破坏性改动必须升版本并加迁移说明。

## 7. Rust vs Go 决策矩阵

| 维度 | Rust | Go |
|---|---|---|
| 与现有 `mq/` 复用 | 高（同工具链/依赖） | 低（重写适配层） |
| 长跑稳定性/内存 | 无 GC 抖动，可预测 | GC 抖动小但存在 |
| 开发速度（本项目） | 中（工具链已就绪） | 未安装，需引入 |
| 生态（系统/协议） | 强 | 强 |
| Web/控制台 | 中 | 强 |
| 结论 | 服务平面默认 | 仅 Web 控制台可选 |

## 8. 风险与回滚

- 迁移期间保持**双入口可对拍**（Python CLI 与 Rust 网关输出 JSON 必须一致），任何不一致阻断发布。
- Python worker 重启必须从 `latest.pt` 恢复且 `config_digest` 匹配，否则拒绝启动（fail closed）。
- 每个阶段独立提交、独立证据；不达标即回滚该阶段，不影响已接受的 Stage 0 资产。
