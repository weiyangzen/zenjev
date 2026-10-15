# TokLens v3：Qwen3.5-0.8B LoRA 蒸馏训练方案（基线版）

**状态：** 训练方案记录 + 基线训练进行中（未做超参优化，后续 HPO 以此为对照）
**日期：** 2026-10-15
**权威说明：** 本文件是研究/方案记录，不改变 `Docs/stage0_zenjev_blueprint.md` 的 checklist 状态；Qwen3.5-0.8B 是本项目钦定的蒸馏目标模型（替代 GLiNER2/205M 一代的文档口径）。

## 1. 目标与边界

把 teacher 模型在 **TokLens schema v6.2 标注任务**上的能力，蒸馏进本地 **Qwen3.5-0.8B**（LoRA SFT）。

- 任务：给定一条 `REQUEST`（用户请求全文）与 `RESPONSE`（助手输出全文），输出 schema v6.2 的 8 个字段：`task_l1/task_l2, domain_l1/domain_l2, artifact_l1/artifact_l2, language, tool`。
- 精度定义（严格口径，训练/评估一致）：输出解析为 JSON dict 后，**8 个 key-value 与 ground truth 完全一致**（键序/字符序无关）；parse 失败或任一字段不一致 = 错误。
- 部署口径：与 teacher 完全相同的输入构造（完整 request/response，64K 窗口）+ 约束解码（闭集枚举 schema + l1-l2 配对校验）。
- 本阶段只做基线 LoRA 训练；**超参数优化（HPO）在后续阶段进行**（见 §9）。

## 2. 数据源（v3）

| 项 | 值 |
|---|---|
| 原始语料 | `192.168.50.220:/mnt/data/toklens-1005merge-{gpt-5.5, gpt-5.6-sol, gpt-5.6-terra, gpt-6-astra, gpt-6-sol}` |
| 语料规模 | **11,251,660** 条（4,259,071 / 4,029,747 / 2,894,369 / 68,469 / 4） |
| 抽样 | 每文件夹 **0.2%**，reservoir 抽样，固定种子 `toklens-0.2pct-sampling-2026-10-13` |
| 抽样结果 | **22,504** 条（8,518 / 8,059 / 5,789 / 137 / 1），8.4 GB |
| 留痕 | 220：`~/sample02-lists/*.txt`、`~/sample02_files.txt`、`~/sample02_reservoir.py`、`~/sample02_seed.bin`；38：`~/toklens-1005label/gt-v3-lists/` |
| 传输 | 220→38 `rsync --files-from`（密钥 `~/.ssh/id_ed25519_38`）→ `38:~/toklens-1005label/sample-v2/` |

语料形态：GPT 系旗舰模型（5.5 / 5.6-sol / 5.6-terra / 6-astra / 6-sol）的真实抓取记录，含 `input_body`（instructions + 全部轮次 + tool calls/results）与 `output_body`（流式响应）。

## 3. GT 生成（teacher 标注）

- **Teacher**：`teacher 模型`（draft 加速，8 draft tokens），SGLang，端口 30000，**单并发**（`max-running-requests 1`）。
- **稳定服务配置（64K）**：`mem-fraction-static 0.94` / `max-mamba-cache-size 12` / `max-total-tokens 64000` / `context-length 65536` / fp8 KV / flashinfer / 关 prefill+decode CUDA graph / chunked-prefill 1024 / draft 加速。
  - 实测：pool=64000 满额分配；63K 探针 12.9s（≈4.9K tok/s prefill）。
  - **失败留痕**：80K 配置（mem 0.96 + mamba 4）在持续负载下因 mamba cache 槽耗尽崩溃（`AssertionError: Can not alloc mamba cache`），已弃用并存档 `watchdog_server.sh.maxctx-80k-failed-20261013`。
  - **坑**：`--max-total-tokens` 是声明值，实际分配须读 `/get_server_info`（mem 0.88 时声明 64K 实际只有 18,654）。
- **输入构造**（`label_client_full.py`）：
  - system = v6.2 树 + 标注指令（2,844 tokens，teacher tokenizer 计量）；
  - user = `REQUEST:\n{完整请求}\n\nRESPONSE:\n{完整响应}`；
  - 预算：user 内容窗 **60,356 tokens**（= 63,400 raw − system − 200 余量）；
  - **RESPONSE 100% 全量（无配额）**；**REQUEST 上限 16,384（≈p51）+ 回吃（backfill）**：resp 用不完的空间全部给 req；
  - 超限侧按 **head 35% / tail 65%** 截断，中间插 `[... truncated ...]` 标记（尾部保留最新用户要求 + 最新上下文）。
- **约束解码**：`response_format json_schema`（`label_schema_v6.2.json`，name=`toklens_v62`，xgrammar 强制）+ 客户端 JSON/l1-l2 配对校验 + 非法重试 ≤5（hint 拼接）。
- **产出**：`labels_v6.2_full.jsonl`，**22,503 ok / 1 failed**（连续 5 次非法配对，按决策不复跑）；全程 2026-10-13 12:39 → 10-15 03:05（≈38.4h），prefill 约 605M tokens @ ~4,924 tok/s。
- **Recipe 留痕**：`38:~/teacher-svc/Docs/GT-REDO-recipe-v3-2026-10-13.md`（+ `~/toklens-1005label/` 副本）。

## 4. 训练集构建（与 GT 同口径）

- `build_dataset_v3.py`：用与 GT 完全相同的渲染 + 截断逻辑复现，**token_mismatch = 0**（22,503 条逐条与 GT 发送的 req/resp token 数一致）。
- 统计：req 截断 6,172 条（**27.4%**）；resp 截断 **0** 条；one_shot（req 或 resp 缺失侧）9,015 条（40.1%）。
- 切分：**train 21,585 / val 459 / test 459**（96/2/2，分层键 = source × request_mode × response_mode × task_l1，seed 42）。
- 格式：ms-swift messages JSONL（system / user / assistant）；`eval_prompts.jsonl` 918 条（val+test 预编码，swift `qwen3_5` 模板，截到 assistant 生成点）。
- assets：`label_schema_v6.2.json`（扁平枚举）/ `l2_children_v6.2.json` / `v6.2_tree.md`。
- 产出路径：`38:/home/sansha/qwen35-0.8b/datasets/toklens-v6.2-v3/`。

## 5. 模型

- **Qwen3.5-0.8B**（`/workspace/models/Qwen3.5-0.8B`）：~**0.873B** 参数（1.75 GB bf16）。
- 架构 `Qwen3_5ForConditionalGeneration`（多模态壳；文本主干 hidden 1024 / 24 层 / 8 heads / vocab 248,320；native ctx 262,144；GatedDeltaNet 混合层）。
- 训练/评估/部署统一用该 checkpoint；LoRA adapter 为唯一可训练参数。

## 6. 训练配置（当前基线，LoRA）

| 项 | 值 |
|---|---|
| 方法 | PEFT LoRA：r=32 / α=64 / dropout 0.05 / `all-linear` |
| 优化 | lr 1e-4（cosine）/ warmup 3% / 3 epochs = **4,050 步** |
| 批 | bs 1 × grad-accum 16（有效 16） |
| 序列 | `max_length 65,536`（同 64K 口径）/ truncation delete / group_by_length |
| 显存技巧 | gradient checkpointing / bf16 / flash-attn 2.8.3 / torch 2.13+cu130 |
| 随机 | seed 42 / data_seed 42 |
| 硬件 | RTX 5090 32GB（192.168.50.38），容器 `qwen35-train`（ms-swift） |
| 实测 | 显存 15.3 GB；稳态 **44–55 s/优化步**（≈3.4 s/样本，~9K tok/s）；3 epochs ≈ 1.9B tokens |
| 运行目录 | `38:~/qwen35-0.8b/outputs/toklens_v63_lora/v1-20261015-041646` |
| 启动 | 2026-10-15 04:16；ETA ≈ 1.8–2.5 天 |

## 7. 评估协议（oracle；HPO 必须固定不变）

- **流程**（`eval_accuracy.py`，vLLM + xgrammar）：约束解码（扁平闭集枚举）→ JSON 解析 → l1-l2 配对校验 → 失败原地重试 ≤5（hint 插在最后一个 `<|im_end|>` 前）。
- **指标**：`exact_match`（8 字段 dict 全等；**键序/字符序不参与比较**）、`first_shot_exact`、`pairing rate`、per-field acc、attempts 直方图、one_shot 子集精度。
- **范围**：checkpoint 评估 = **val 前 100 条固定子集**（`watch_eval.sh`，显存守护，约 2 分钟/次，每 125 步一个 checkpoint）；终评 = val 459 / test 459 全量。
- **工作台**：`http://192.168.50.38:8899`（loss / token_acc / lr 曲线 + acc 曲线 + 评估明细，30s 自动刷新）。
- **当前曲线（val-100，进行中）**：

| ckpt | exact | task_l1 | task_l2 | domain_l2 | artifact_l2 | tool |
|---|---|---|---|---|---|---|
| step-3（基线） | 0% | .35 | .10 | .14 | .09 | .67 |
| step-125 | 18% | .64 | .41 | .67 | .66 | .75 |
| step-250 | 29% | .76 | .65 | .76 | .78 | .88 |
| step-375 | 27% | .77 | .68 | .75 | .79 | .81 |
| step-500 | 31% | .80 | .73 | .72 | .82 | .85 |

（实时值：step 600 时 train_loss 0.051 / token_acc 98%；exact 为 8 字段联乘口径，单字段 72–85% 时 exact ≈ 0.3 属正常。）

## 8. 基线参照

| 运行 | 数据口径 | 方法 | 结果 |
|---|---|---|---|
| v1 full-FT（2026-10-12） | 旧 0.5% 样本 / 8K 截断口径 | 全参 SFT，10,110 步 / 10.8h | val **38.77%** / test **40.07%** |
| v3 LoRA（本方案，进行中） | 0.2% 样本 / 64K 同口径 | LoRA r32，4,050 步 | 见 §7 曲线（step-500 = 31%） |

## 9. 超参优化（HPO）计划锚点

**本基线未做任何超参搜索**；后续 HPO 要求：

- **可调维度（候选）**：lr {5e-5, 1e-4, 2e-4}；LoRA r {16, 32, 64} 与 α（1r/2r）；dropout {0, 0.05, 0.1}；target modules（all-linear vs 仅 attention/mlp）；epochs {1, 2, 3}；有效 batch {8, 16, 32}；scheduler/warmup；weight decay；packing 开/关；截断策略（quota/head-tail 比例）与 max_length。
- **方法**：固定 oracle（同一 val-100 子集 + 同一约束解码评估流程），先用**子集短跑**（如 train 子集 × 1 epoch）粗筛，再对优胜组合全量复验；每次实验记录 loss 曲线、token_acc 与 exact 曲线。
- **纪律**：评估子集与 GT 固定；**任何数据口径/渲染/截断改动 = 新数据版本**，不得混入同组对照；重跑基线须用同 seed 复现对照。
- **预算**：单卡 5090；一轮 3-epoch 全量 ≈ 1.9B tokens ≈ 1.8–2.5 天 → HPO 必须用小规模协议控制成本。

## 10. 复现资源清单

| 资源 | 位置 |
|---|---|
| 采样脚本/列表/种子 | 220 `~/sample02_reservoir.py`、`~/sample02-lists/`；38 `~/toklens-1005label/gt-v3-lists/` |
| GT 客户端 | `38:~/qwen35-0.8b/tools/toklens_gt/label_client_full.py`（+ `prep_stats.py`） |
| Teacher 起服/守护 | `38:~/teacher-svc/watchdog_server.sh`（+ 80K 失败版存档） |
| GT recipe | `38:~/teacher-svc/Docs/GT-REDO-recipe-v3-2026-10-13.md` |
| 建库 | `38:~/qwen35-0.8b/tools/toklens/build_dataset_v3.py`、`build_eval_prompts_v3.py` |
| 训练 | `38:~/qwen35-0.8b/scripts/run_sft_lora_v3.sh`（容器内 `bash /workspace/scripts/run_sft_lora_v3.sh`） |
| 评估/守护/面板 | `38:~/qwen35-0.8b/tools/toklens/{eval_accuracy.py, watch_eval.sh, dashboard.py}` |
| 数据集 | `38:~/qwen35-0.8b/datasets/toklens-v6.2-v3/`（train/val/test + meta + eval_prompts + stats） |
| GT 标签 | `38:~/qwen35-0.8b/toklens_gt02/out/labels_v6.2_full.jsonl` |
| 机器 | `.38` 训练+教师；`.220` 语料；`.185` 备份 NAS |
