# TokLens v6.2：GT / 训练 / 评估的输入输出格式说明

**状态：** 格式规范 + 数据缺陷记录（v3 现状；v3.1 修复 / v3.2 剔除两侧皆空进行中，见 §7–§8）
**日期：** 2026-10-16
**权威说明：** 本文件是研究/方案记录，描述 `toklens-v6.2-v3` 数据链路的实际格式与已知缺陷；不改变 `Docs/stage0_zenjev_blueprint.md` 的 checklist 状态。配套方案见 `toklens_qwen35_08b_v3_training_plan.md`。

## 1. 结论先行

- 输入**不是** `{"req": ..., "resp": ...}` 这种结构化字段，而是**一段纯文本**：
  `REQUEST:\n{渲染后的请求}\n\nRESPONSE:\n{渲染后的响应}`，缺失侧写字面量 `(missing)`。
- **GT 标注（teacher）与学生训练/评估的输入逐字节相同**：`build_dataset_v3.py` 直接 `import label_client_full` 复用同一套渲染与截断函数，并逐条校验发送 token 数（`token_mismatch = 0`）。
- 输出是 8 个字段的扁平 JSON（闭集枚举），GT 与学生同 schema。

## 2. 原始抓取（capture）的两种 schema

抽样文件：`38:~/toklens-1005label/sample-v2/<source>/<file>.json`（容器内 `/toklens/sample-v2/`，只读）。

| schema | 请求体字段 | 响应体字段 | 覆盖 |
|---|---|---|---|
| 主 schema（`capture_key` / `redacted` 等） | `input_body`（OpenAI Responses API 请求 JSON 字符串） | `output_body`（SSE 流文本，或完整 response JSON） | 绝大多数 |
| 备用 schema（`request_json` / `response_json` / `content_text` 等） | `request_json`（同为 Responses API 请求 JSON 字符串） | `response_json`（完整 response JSON） | 仅 `toklens-1005merge-gpt-5.5`，抽样中 **350 条** |

抓取侧已知的内容丢失形态：

- `input_body = "[omitted:request_body_too_large len=<bytes> sha256=<16hex>]"`：请求体超过约 0.5–1 MB 时采集端只留占位符，**原文不在语料中**。
- `input_body` 为 JSON 但只有 `{"_capture_truncation": ..., "model": ...}`：采集端截断，无可渲染内容。
- `input_body` / `output_body` 字段缺失（`None`）。

## 3. 渲染规则（`tools/toklens_gt/label_client_full.py`）

### 3.1 取体

`capture_bodies(d)` → `(d.input_body or d.request_json, d.output_body or d.response_json)`（2026-10-16 加入回退，见 §7）。

### 3.2 REQUEST（`render_request`）

解析请求 JSON，按顺序拼接块，块之间以空行（`\n\n`）分隔：

| 来源 | 渲染 |
|---|---|
| `instructions`（非空字符串） | `SYSTEM:\n{instructions}` |
| `input` 为字符串 | `USER:\n{input}` |
| `input[]` 中 `type ∈ {message, None}` 且 `role ∈ {user, assistant, system, developer}` | `{ROLE}:\n{文本}`（文本取 content 中 `text / input_text / output_text / content`） |
| `function_call / custom_tool_call / local_shell_call / tool_call` | `TOOL_CALL {name}:\n{arguments 或 input（非字符串则 json.dumps）}` |
| `function_call_output / custom_tool_call_output / local_shell_call_output / tool_result` | `TOOL_RESULT:\n{output 或 content}` |
| `reasoning`（有 summary 文本） | `REASONING:\n{summary 文本}` |

占位符 `[omitted...]` 或空 → 空串；非 JSON 字符串 → 原文（截 2M 字符）。

### 3.3 RESPONSE（`render_response`）

- 完整 response JSON：逐个 `output[]` item 渲染后以 `\n` 连接；message → 文本；tool call → `[tool_call {name}] {args}`；tool result → `[tool_result] {output}`；reasoning → summary 文本。无 item 时退回 `output_text / text`。
- SSE 流：只取 `response.output_text.delta`、`response.reasoning_summary_text.delta`、`response.function_call_arguments.delta`、`response.custom_tool_call_input.delta` 的 delta，以及 `response.output_item.done` 的 item 渲染，**直接拼接**（无分隔）。

### 3.4 预算与截断（teacher tokenizer：`/workspace/teacher_tok/tokenizer.json`）

- 总预算 `MAX_INPUT_TOKENS = 63,400`（system + user 原文）；system = 2,844 tokens；模板余量 200 → **user 内容窗 60,356**。
- **RESPONSE 优先全量**（无配额）；REQUEST 先限 `REQ_QUOTA = 16,384`，开启 `REQ_BACKFILL=1` 时回吃 RESPONSE 用剩的全部空间：
  ```
  r_b = min(req_need, 16384); p_b = min(resp_need, 60356 - r_b); r_b = min(req_need, 60356 - p_b)
  ```
- 超限侧按 **head 35% / tail 65%** 截断，中间插入 `\n[... truncated ...]\n`。

### 3.5 最终 user 文本

```
REQUEST:
SYSTEM:
You are Codex, a coding agent ...

USER:
<用户消息>

TOOL_CALL exec_command:
{"cmd": "..."}

TOOL_RESULT:
<工具输出>
[... truncated ...]
<尾部 65%>

RESPONSE:
<回复文本>[tool_call apply_patch] {...}
```

## 4. system 与输出

- system = `build_system(v6.2_tree.md)`：TokLens v6.2 闭集标注指令 + l1-l2 归属规则 + 缺失侧规则（"request 缺失则只看 response，反之亦然"）+ 工具驱动样本规则 + 截断说明 + 完整 taxonomy 树，结尾 `Output JSON only.`
- 输出（assistant）= 8 字段扁平 JSON，键序固定：
  ```json
  {"task_l1": "...", "task_l2": "...", "domain_l1": "...", "domain_l2": "...",
   "artifact_l1": "...", "artifact_l2": "...", "language": "...", "tool": "..."}
  ```
  约束：`*_l2 ∈ l2_children[*_l1]`；`language` / `tool` 为闭集枚举（`assets/label_schema_v6.2.json`、`assets/l2_children_v6.2.json`）。

## 5. 落盘格式

| 产物 | 路径（38） | 行格式 |
|---|---|---|
| GT 标签 | `~/qwen35-0.8b/toklens_gt02/out/labels_v6.2_full.jsonl` | `source, file, request_id, created_at, model_source, request_mode(full/empty), response_mode, req_tokens_full, resp_tokens_full, sent_req_tokens, sent_resp_tokens, truncated_request, truncated_response, prompt_tokens, completion_tokens, labels{8 字段}, attempts, elapsed_s, ts` |
| 训练/验证/测试 | `~/qwen35-0.8b/datasets/toklens-v6.2-v3/{train,val,test}.jsonl` | `{"messages": [system, user, assistant]}`（ms-swift messages） |
| 行元数据 | 同目录 `meta.jsonl` | `split, source, file, request_mode, response_mode, task_l1, sent_req_tokens, sent_resp_tokens, truncated_*, one_shot`；**行序与三个 split 文件一一对应** |
| 评估预编码 | 同目录 `eval_prompts.jsonl` | `split, idx(=split 内行号), src, file, one_shot, input_ids(swift qwen3_5 模板，止于 assistant 生成点), gt{8 字段}` |

训练模板：ms-swift `qwen3_5`，`add_non_thinking_prefix=True`，loss 只算 assistant（`loss_scale=default+ignore_empty_think`）。

## 6. 缺失分布（v3 GT 全量 22,503 条，2026-10-16 逐条扫描）

| 类别 | 条数 | 原因 |
|---|---|---|
| REQUEST 空：采集占位符 `request_body_too_large` | 4,271 | 抓取端丢弃，抽样文件中无原文 |
| REQUEST 空：`input_body` 缺失 | 3,992 | 抓取端无请求体 |
| REQUEST 空：`_capture_truncation` 残体 | 81 | 抓取端截断 |
| **REQUEST+RESPONSE 空：备用 schema 未读取** | **350** | **管线缺陷，见 §7** |
| RESPONSE 空 | 1,522（其中 1,201 两侧皆空） | 抓取端无响应体 / 上述 350 |

两侧皆空的 1,201 条，teacher 在空输入上只给出 2 种标签（约 59/41），属同输入矛盾标签，学生无法学会。

**评估子集偏差**：v3 `val` 前 100 条（训练期 checkpoint 曲线所用）中 72% 为 REQUEST 缺失、8% 两侧皆空、仅 19% 两侧完整；全量 val 459 条中两侧完整占 60%（274/459）。前 100 条口径的 exact 不代表全量分布，HPO 与最终结论以全量 val 459 为准，并按类别（full / req_missing / both_missing / resp_missing）分报。

## 7. 缺陷：备用 schema 的 350 条（v3）与 v3.1 修复

- **现象**：`toklens-1005merge-gpt-5.5` 中 350 条抓取的请求/响应位于 `request_json` / `response_json`，v3 渲染只读 `input_body` / `output_body` → 两侧均被渲染为 `(missing)` → teacher 在空输入上标注，**350 条 `task_l1` 全部为 `other`**。
- **实际内容**：用现有渲染函数读取备用字段后 350/350 均可正常渲染；REQUEST 中位 96,465 tokens（将被截断到预算内），RESPONSE 中位 148 tokens。
- **分布**：train 334 / val 8 / test 8。
- **修复**（2026-10-16）：
  1. `label_client_full.py` 新增 `capture_bodies()`（字段回退），`process_one` 与 `build_dataset_v3.py` 改用之（原文件备份 `*.bak-20261016`）。
  2. `hpo/relabel_alt350.py`：同 teacher（同配置标注服务）、同 system、同约束 schema、同预算（quota 16,384 + backfill）只重标这 350 条 → `toklens_gt02/out/labels_v6.2_altfix.jsonl`；**原 GT 文件不改**。
  3. `hpo/build_dataset_v31.py` → `datasets/toklens-v6.2-v3.1/`：**split 成员与行序完全沿用 v3 `meta.jsonl`**（重跑分层切分会因 task_l1/request_mode 变化而整体洗牌）；其余 22,153 条与 v3 逐字节一致（构建时断言；抽样 632 条预检 0 差异），仅替换 350 条；`meta.jsonl` 增加 `altfix` 标记。
## 8. v3.2：剔除两侧皆空（HPO v2 与最终训练所用数据集）

- **定义**：v3.2 = v3.1 − 两侧皆空行（`request_mode == response_mode == empty`）；v3.1 未就绪时回退为 v3 − 全部两侧皆空行（含 350 条错标）。split 内行序保持不变。构建：`hpo/build_dataset_v32.py`，评估预编码：`hpo/build_eval_prompts_v32.py`；`eval_accuracy.py` 新增 `--ds` 指定数据集（默认仍为 v3）。
- **规模**（以 v3.1 为基）：剔除 train 819 / val 16 / test 16 → **train 20,766 / val 443 / test 443**。
- **为什么剔除**：
  - 训练侧影响小：两侧皆空只占 train 行数 3.8%、训练 token 0.53%，同一输入的两种标签只在 `language` 上不同（other 505 / en 346）。
  - 评估侧是纯噪声：基线 cosine run 在 v3 val 的 24 条两侧皆空上，step-625/750/1000/1625 均为 70.8%（输出多数标签），step-875 翻到少数标签降为 29.2%，使整体 exact 凭空下降约 2.2pp；而 HPO 各组比较的恰是数个 pp 的差距。
  - 部署侧：两侧皆空的输入由规则直接处理，不经过模型。
- **口径**：HPO v2 全部 trial、锚点（CD0625，v3 训练、v3.2 val 评分）与最终训练均使用 v3.2；最终报告同时给出 v3.2 与 v3 val/test 的 exact。

## 9. 代码与路径索引（38，容器 `qwen35-train`，`/workspace = ~/qwen35-0.8b`）

| 用途 | 路径 |
|---|---|
| GT 标注客户端（渲染/截断/约束解码/校验） | `tools/toklens_gt/label_client_full.py` |
| 训练集构建 | `tools/toklens/build_dataset_v3.py` |
| 评估预编码 | `tools/toklens/build_eval_prompts_v3.py` |
| 约束解码评估（exact / pairing / field_acc / 逐样本） | `tools/toklens/eval_accuracy.py` |
| 350 条重标 / v3.1 构建 / v3.2 构建 | `hpo/relabel_alt350.py` / `hpo/build_dataset_v31.py` / `hpo/build_dataset_v32.py` |
| Schema 资产 | `datasets/toklens-v6.2-v3/assets/`（`label_schema_v6.2.json`、`l2_children_v6.2.json`、`v6.2_tree.md`） |
