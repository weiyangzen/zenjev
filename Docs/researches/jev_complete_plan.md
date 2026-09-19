# Jev：基于 GLiNER2 205M + LoRA/EMA 的可配置 Schema 抽取系统研究方案

**状态：研究方案（Stage 0 输入，不是运行时实现）**  
**更新：2026-09-20**

本文给出 Jev 的模型、蒸馏、持续训练、在线推理、漂移检测和 5090 部署方案。`Docs/stage0_zenjev_blueprint.md` 是执行阶段的唯一权威；本文件记录设计依据、可验证的决策和风险。

## 1. 目标与边界

Jev 接收用户定义的 Schema 和用户允许的数据源，将可信教师模型的结构化标注蒸馏为本地 GLiNER2 数据；本地 205M 编码器用 LoRA 做增量训练。推理服务一直使用经过验证的模型快照，训练服务可并行更新 LoRA 和其 EMA（exponential moving average）影子权重。新快照只有通过固定验证集、回归集和漂移门禁后才原子切换。检测到 LoRA 累计误差崩溃时，删除该适配器和优化器状态，从冻结的基础模型重新训练。

范围包括：

* Schema 的版本化、字段描述、类型、约束、语言和来源配置；
* 可插拔教师提供商（例如用户已获授权的 GPT 系列或 DeepSeek 系列端点），不把某个未来模型名称写死；
* URL/RSS/API/本地文件等经过允许列表的数据源，抓取、去重、证据保留和审计；
* GLiNER2 205M 本地抽取、LoRA 适配器、EMA、在线推理和回滚；
* 在 RTX 5090 主机上可复现的训练和基准门禁。

不把教师模型的输出当作事实真值：教师结果必须经过 JSON Schema、证据引用、重复样本和人工/规则门禁后才能进入训练集。

## 2. 证据和模型选型

### 2.1 GLiNER2

Fastino 的官方模型卡明确将 `fastino/gliner2-base-v1` 标为 **205M 参数、DeBERTa-v3-base、English、span 架构**，并给出 `extract_entities`、分类和结构化 JSON 的 Schema 驱动接口；仓库的模型表和训练教程也给出该 checkpoint 的 LoRA 配置和 adapter-only 保存方式。

* 模型卡：[fastino/gliner2-base-v1](https://huggingface.co/fastino/gliner2-base-v1)
* 官方实现、Schema 和训练教程：[fastino-ai/GLiNER2](https://github.com/fastino-ai/GLiNER2)
* 论文：[GLiNER2: An Efficient Multi-Task Information Extraction System with Schema-Driven Interface](https://arxiv.org/abs/2507.18546)

默认架构选择 `fastino/gliner2-base-v1`，冻结全部基础参数，只在 encoder/task heads 的 LoRA target 上训练。它的接口适合用户在运行时传入实体、分类、结构化记录和关系 Schema。不要在 Stage 0 把 `gliner2.5-base-v1`（约 194M、boundary 架构）误写成用户要求的 205M GLiNER2；若后续需要任意长 span，应另建经过评测的模型变体。

用户示例含中文（“2026 最佳技术栈”）。`gliner2-base-v1` 的公开训练语言是 English；需要中文或混合语言时，配置选择官方 `fastino/gliner2-multi-v1`（约 205M、mDeBERTa-v3-base）并在同一评测协议下比较。GLiNER2 文档还指出 word splitter 必须与训练时一致；中文数据需使用 `char` splitter 或用多语 checkpoint 重新验证，不能只切换分词器后宣称效果等价。

### 2.1.1 `jev-tool-task`：从任务类型到受控路由

GLiNER2 的官方 Schema API 支持文档分类、结构化记录和关系抽取，因此适合把 request/response 落地池压缩成有限的决策特征：request 侧用 `task_type` 分类和 `task_detail` 实体/字段，response 侧用 `tool`、`programming_language`、`framework`、`technology`、`model` 实体以及 `uses_*` 关系。官方教程分别覆盖[分类](https://github.com/fastino-ai/GLiNER2/blob/main/tutorial/1-classification.md)、[JSON/record 抽取](https://github.com/fastino-ai/GLiNER2/blob/main/tutorial/3-json_extraction.md)、[组合 Schema](https://github.com/fastino-ai/GLiNER2/blob/main/tutorial/4-combined.md)和[关系抽取](https://github.com/fastino-ai/GLiNER2/blob/main/tutorial/6-relation_extraction.md)。

这不等于模型可以根据任务类型自由生成并执行决策。GLiNER2 是 schema-conditioned extractor/classifier：它可以在预先定义的有限标签集合中输出 `task_type`、`decision_route`、confidence 和 evidence，但不会成为规划器或工具调用器。Jev 需要在 `configs/jev_tool_task.yaml` 中维护模型、工具、编程语言和技术栈 allowlist；先依据落地记录中的可信 `observed_model` 做精确匹配，再抽取 request 和 response，最后由 `jev.tool_task.resolve_tool_task` 做置信度、证据、冲突和 allowlist 检查。未知模型直接 reject；低置信度或缺少证据进入 review/fallback；只有 policy 输出 `allow` 后，独立工具适配器才有资格执行。

推荐的标准记录是：`request_id`、source URI/hash/retrieved_at/license、observed provider/model、`request.text`、`response.text`。request/response 分开推理，避免 response 泄漏到任务类型；同一个 schema digest 和 adapter/EMA metadata 写入两侧结果。若启用 GLiNER2 分类约束 DSL，应锁定安装版本并增加 implies/excludes 的正负 golden tests；约束只缩小候选标签，Jev policy 仍是最终安全边界。当前 pinned 205M span checkpoint 不应假设 GLiNER2.5 boundary 的 JointIE typed graph 能力。

### 2.2 LoRA

LoRA 在冻结基础权重上注入低秩可训练矩阵，减少显存、优化器状态和 checkpoint 体积；原始论文是 [LoRA](https://arxiv.org/abs/2106.09685)，可执行 API 和 target-module 约定见 Hugging Face [PEFT LoRA conceptual guide](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora)。GLiNER2 官方训练教程提供了可复现实例：`use_lora=True`、`lora_r=8`、`lora_alpha=16`、`save_adapter_only=True`，并示例将 encoder 与所有 task heads 作为 target。Jev 初始值采用 `r=8, alpha=16, dropout=0.05`，但这些是待基准验证的超参，不是质量保证；Stage 0 应至少比较 `r=8` 与 `r=16`。

适配器 artifact 必须包含：基础模型 ID 和 revision、LoRA config、Schema ID/version、数据切分哈希、训练配置、代码版本、随机种子、训练步数、验证指标和 SHA-256。只允许加载与当前基础 checkpoint hash 匹配的 adapter。

### 2.3 EMA

EMA 不是额外的教师模型。它是 LoRA 可训练参数的影子副本，在每个成功优化步后更新：

```text
ema_t = decay * ema_(t-1) + (1 - decay) * live_lora_t
```

基础模型保持冻结，不需要为其维护 EMA。推理优先使用最近一次通过门禁的 EMA adapter；训练继续使用 live adapter。可用 `decay=0.995`（短窗口）或 `0.999`（长窗口）起步，按验证集延迟和质量调节。更新和导出必须在单线程/单进程临界区内完成；导出到临时目录、fsync、校验 manifest 后，再用原子 rename/符号链接切换。PyTorch 的参数平均抽象见 [`torch.optim.swa_utils.AveragedModel`](https://docs.pytorch.org/docs/stable/generated/torch.optim.swa_utils.AveragedModel.html)；生产实现也可参考 timm 的 [`ModelEmaV2`](https://github.com/huggingface/pytorch-image-models/blob/main/timm/utils/model_ema.py)。Jev 需要明确记录 decay、更新步数和 EMA checkpoint 的来源，避免把 EMA 当作无验证的“自动更好”。

## 3. 运行时架构

```text
                ┌────────────────────────────────────┐
user request ──▶│ inference worker                      │
                │ frozen GLiNER2 + active EMA adapter │──▶ structured result
                └───────────────▲────────────────────┘
                                │ atomic promote / rollback
                                │
source config ─▶ fetch/normalize ─▶ teacher adapter ─▶ validator ─▶ replay buffer
                                                                │
                                                                ▼
                ┌──────────────────────────────────────────────────────────────┐
                │ trainer worker (5090)                                         │
                │ live LoRA optimizer ─▶ EMA update ─▶ holdout/eval ─▶ candidate │
                └──────────────────────────────────────────────────────────────┘
```

* **Inference worker**：加载一个 immutable base + active adapter，按 Schema 运行；每个请求记录 schema/version、model/adapter hash、阈值、来源和时间。它不在请求线程内修改权重。
* **Trainer worker**：从已验证 replay buffer 取小批次，更新 live LoRA，再更新 EMA；周期性在冻结 holdout/canary 上评测。评测和导出使用独立 eval 实例，禁止 eval 线程观察半写入权重。
* **Promotion controller**：验证 candidate 的 manifest 和指标，写入 `adapters/<id>/` 后原子更新 `active.json`；失败时保留旧 active。每次切换都可审计和回滚。
* **Reset controller**：执行第 6 节的硬门禁。重置期间 inference 继续服务旧 active；新 adapter 只有通过门禁才可上线。

同一进程共享 GPU 权重会让训练、EMA 和推理产生竞态；首版应使用独立 inference/trainer 进程或严格的读写锁与 copy-on-write。5090 显存足以容纳一个 205M fp16 基础模型和 LoRA/EMA 副本，但不能假设无限余量；启动时记录 `torch.cuda.mem_get_info()`，OOM 时退回 bf16/fp16、较小 batch 和 gradient accumulation。

## 4. 用户 Schema 合约

Schema 是用户配置而不是模型权重，必须版本化且可回放。建议采用 JSON/YAML 配置，字段如下：

```yaml
id: best_tech_stack
version: 1
language: zh-CN
model_preference: fastino/gliner2-multi-v1
splitter: char
entities:
  technology: "软件、框架、数据库、云服务或硬件产品名称"
  vendor: "提供该技术的组织"
  use_case: "技术被使用的场景"
classifications:
  maturity: [production, beta, experimental]
record:
  name: recommendation
  fields:
    technology: {type: string, required: true}
    category: {type: string, required: true}
    rationale: {type: string, required: true}
    evidence: {type: array[string], required: true}
    as_of: {type: date, required: true}
constraints:
  evidence_min: 1
  require_source_url: true
  reject_unverifiable_claims: true
```

“2026 最佳技术栈”不是一个无条件的标签。应把年份、任务目标、约束和证据放入 Schema，输出每项 recommendation 的证据句/URL 和 `as_of` 日期；教师只从用户授权来源生成候选，不凭参数记忆补全。Schema 变更必须递增版本并重新生成/验证样本，旧 adapter 不自动用于新版本。

服务端在接受 Schema 时检查：唯一字段名、允许的类型、最大长度、枚举值、关系端点、语言与 splitter、prompt 模板版本以及来源允许列表。输出先通过 JSON Schema 验证，再做 span/证据存在性验证；无效样本进入 quarantine，不进入 replay buffer。

## 5. 可配置来源和蒸馏流水线

### 5.1 来源配置

来源由用户显式配置并存储 manifest。每个 source 至少包含：`id`、`kind`（http/rss/api/file）、URL/路径、允许域名、抓取时间窗、认证引用（只存 secret 名称）、robots/许可策略、最大字节数、更新周期、内容选择器、语言和去重键。抓取原文保存在对象存储或本地内容寻址目录，计算 SHA-256；输出保留 source URL、发布时间、抓取时间和原文偏移，方便审计和回放。

建议顺序：fetch → MIME/大小/域名检查 → HTML/文档规范化 → 语言检测 → canonical URL 与内容哈希去重 → 文本分块（保留重叠和偏移）→ teacher 请求 → JSON Schema/证据校验 → train/validation/test 按文档哈希分组切分。永远按文档或站点分组切分，不能把同一页面的相邻块随机分到 train 和 holdout。

### 5.2 教师 provider 接口

```text
TeacherProvider.generate(schema, document, source_metadata, request_id)
  -> raw_response, usage, provider_model, timestamp
```

配置包含 provider 类型、模型名、endpoint、超时、重试/退避、速率限制、最大 token、随机性（蒸馏默认低 temperature）、密钥引用和成本上限。模型名只作为配置字符串；如果用户有权使用 GPT-6 Astra、DeepSeek 4.1f 或其他 provider，直接实现该接口即可。不要在代码中假设这些名称一定存在，也不要把密钥写进 Schema、日志或样本。

教师 prompt 应要求：只输出 Schema 允许的 JSON；为每个值给出原文证据句或字符区间；未知值为 null/空数组；不要添加字段；返回 schema/version。validator 检查：

1. JSON 可解析且符合 Schema 类型/枚举；
2. 每个非空值可在原文或声明的证据片段中定位（规范化前后偏移可追踪）；
3. URL 属于允许来源且来源 hash 存在；
4. 同文档重试结果或双教师结果不一致时进入审查队列；
5. PII/版权策略允许保存和训练，不能把拒绝样本静默丢掉。

蒸馏论文背景见 Hinton 等人的 [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)。Jev 不把蒸馏损失直接假设成 GLiNER2 的训练 API；对 GLiNER2，首先将通过验证的教师结构化结果转换成官方 JSONL/InputExample，再用其 `ExtractorTrainer` 和 LoRA recipe 训练。保留 `teacher_raw_hash` 和验证决策，确保样本可追溯。

### 5.3 数据质量与采样

replay buffer 分成 `recent`、`stable`、`canary` 三层：recent 反映新来源，stable 防止遗忘，canary 是永不被训练覆盖的固定回归集。每个训练 batch 至少混合 stable 与 recent；同一个文档版本只能出现一次。用当前 EMA 的低置信度、字段缺失、教师 disagreement 和新来源域分布做主动采样，但不能以模型自己的错误标签闭环自我强化。

## 6. LoRA 累计误差崩溃判定和重置

“崩溃”必须依据 live/EMA candidate 相对于**冻结基线**和当前最佳 active 的指标，而非只看训练 loss。每次评测使用固定 holdout、canary 和最近窗口，报告 micro/macro F1、字段级 precision/recall、exact JSON validity、evidence grounding rate、ECE（或可靠性分箱）、拒答率、延迟和 adapter norm。

**Stage 0 的冻结判定以 `Docs/stage0_zenjev_blueprint.md` 为准**，研究文档不能覆盖它：

* NaN/Inf 出现在权重、梯度或 loss 时立即停止并记录 terminal failure；
* 其余信号（实体/span F1 相对冻结 baseline 绝对下降至少 0.15 或相对下降至少 0.20、validation loss 超过 baseline 的 2 倍、invalid span/schema output、或配置的 confidence/coverage floor）必须连续 `consecutive_windows=3` 个评测窗口成立才触发 reset；
* 每个窗口至少 256 条样本（不足则标低置信度），按 Schema/语言分别保存 baseline、窗口计数和阈值。

为了提高候选发布质量，可以在不改变上述 reset 语义的前提下配置额外的 promotion 门禁，例如 JSON validity < 95%、evidence grounding < 90%、adapter L2 norm/初始化 norm > 3、ECE > 0.12、teacher disagreement > 20%、验证拒绝率 > 10% 或来源 PSI > 0.25。后两类是数据故障：暂停吸收并 quarantine/复查，不自动擦除质量正常的 adapter。新增门禁必须写入 manifest 和蓝图的策略迁移记录；阈值不是普适常数，应使用 golden set 校准。

**Reset 流程：** (1) promotion controller 立即固定旧 active；(2) trainer 停止接收新 batch，保存诊断和失败 candidate；(3) 从基础模型 revision 重新加载空 LoRA、空 EMA 和新 optimizer；(4) 以 stable + 最近一段已通过验证的 recent 数据 warm-up，使用较小学习率和 early stopping；(5) 在 holdout/canary 上重新评估；(6) 通过 promotion gate 后原子切换，否则继续提供旧 active 并报警。保留最近 N 个 adapter/manifest 供回滚，绝不删除基础 checkpoint 或证据原文。

## 7. RTX 5090 训练/推理计划

NVIDIA 官方产品页：[GeForce RTX 5090](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/) 给出 32 GB GDDR7 显存等硬件规格。部署脚本在启动时记录 GPU 名称、显存、驱动、CUDA、PyTorch、GLiNER2 版本和 git/model revision；“5090 可用”必须由目标主机上的 smoke test 证明，而不是由规格推断。

建议初始配置：

* 基础模型 fp16 或 bf16（以 5090 上的稳定性 smoke test 决定），冻结 base；LoRA `r=8, alpha=16, dropout=0.05`；训练 batch 从 8 开始，OOM 时用 gradient accumulation；
* `torch.autocast`、gradient clipping（例如 1.0）、非有限梯度检测、周期性 EMA export；默认不依赖 FlashAttention 或自定义 CUDA kernel；
* inference 与 trainer 分进程，推理服务优先加载 EMA adapter；仅在 candidate 通过 gate 后 reload；
* 评测基准至少记录 p50/p95 延迟、tokens/s、显存峰值、训练 step/s 和每个 Schema 的质量。以目标 SLO 配置为准，不在研究阶段伪造固定吞吐数字；
* 网络不可用时，推理仍用最近 active；蒸馏队列可暂停并在恢复后按 source hash 去重。

## 8. 质量、可观测性和安全门禁

每个结果的最小审计字段：`request_id`、`schema_id/version`、`source_ids/hashes`、`base_model_revision`、`adapter_id/hash`、是否 EMA、推理阈值、代码版本和时间。训练日志应包含 batch/source 分布、live/EMA 指标、梯度/adapter norm、GPU memory、样本拒绝原因、provider 用量和成本；日志不得包含 API key 或未脱敏 PII。

上线前至少执行：Schema validator 单测、teacher 输出回放、固定 golden/holdout/canary、跨来源去重测试、adapter 原子切换/回滚测试、NaN/OOM/reset 演练、断网推理 smoke test、中文与英文 splitter 对照、5090 GPU smoke test。验收证据应写入 candidate manifest 并由 Stage 0 blueprint 的 gate 引用。

## 9. 已知风险和取舍

* **语言错配**：205M English checkpoint 对中文 Schema 可能失真；必须使用多语 checkpoint 或明确限定语言，并以同一 golden set 验证。
* **伪标签偏差**：教师会把时效性/来源偏差传给学生；证据定位、双采样和 stable/canary 混合只降低风险，不能保证事实正确。
* **在线更新竞态**：直接在 inference 权重上训练会出现半更新读；采用分进程、临时导出和原子 promote。
* **EMA 过度平滑**：decay 过大可能延迟适应；同时评估 live 与 EMA，只有更优且合规的版本才发布。
* **LoRA 容量不足或过拟合**：r、target modules、学习率、recent/stable 比例都需基准；不能由 adapter 大小推断质量。
* **来源许可与隐私**：用户必须确认抓取/蒸馏权利；Jev 只允许配置的 source，保存 provenance 和删除索引。

## 10. 参考资料

1. Fastino, *GLiNER2 model card*（205M、DeBERTa-v3-base、接口和许可证）：<https://huggingface.co/fastino/gliner2-base-v1>
2. Fastino AI, *GLiNER2 source and training tutorials*：<https://github.com/fastino-ai/GLiNER2>
3. Zaratiana et al., *GLiNER2*：<https://arxiv.org/abs/2507.18546>
4. Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*：<https://arxiv.org/abs/2106.09685>
5. Hugging Face, *PEFT LoRA conceptual guide*：<https://huggingface.co/docs/peft/main/en/conceptual_guides/lora>
6. PyTorch, *AveragedModel*：<https://docs.pytorch.org/docs/stable/generated/torch.optim.swa_utils.AveragedModel.html>
7. Hugging Face timm, *ModelEmaV2 implementation*：<https://github.com/huggingface/pytorch-image-models/blob/main/timm/utils/model_ema.py>
8. Hinton et al., *Distilling the Knowledge in a Neural Network*：<https://arxiv.org/abs/1503.02531>
9. NVIDIA, *GeForce RTX 5090 specifications*：<https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/>
