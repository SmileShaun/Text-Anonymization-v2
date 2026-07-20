# Causal llm-anonymization：真实 vLLM Token 统计（300 pers）

用本地 **vLLM + Llama-3.1-8B-Instruct** 跑因果冻结历史匿名化，并记录每个 pers 的 **真实 API usage tokens**（prompt / completion）。

本目录自包含：可整体拷贝到其他机器运行（模型权重路径用 CLI / 环境变量指定）。

| 本地依赖 | 路径 |
|---|---|
| 因果驱动 + `src/` | `vendor/llm-anonymization-v2/` |
| SynthPAI profiles | `data/synthpai/profiles/` |
| 原始 synthpai.jsonl（rebuild / stub） | `data/synthpai/synthpai.jsonl` |
| 300 人列表 | `data/inputs/all300_authors.txt` |

## 流程（每人 / 每条 comment）

- 因果前缀：只见 `fixed_anon[0..M-1] + 当前条`（历史冻结）
- 默认 3 轮：`infer → anonymize → utility`
- **无 25-cap**（full comments）
- 终端进度：`Progress: comments done/total (...)`

## 输出结构

```
results/vllm_llama3.1-8B_all300_tokens/
  pers1/
    token_usage.json    # 逐 comment、逐 call 明细
    result.json
    ...
  pers2/
    ...
  per_user_token_usage.csv      # 跑完自动/手动汇总
  token_usage_summary.json
```

每人目录下 token 明细（`schema_version=2`）：

| 文件 | 内容 |
|---|---|
| `token_usage.json` | 嵌套：comment → round → 每次 LLM 请求（**每完成一条 comment 覆盖更新**） |
| `token_usage_flat.json` | 扁平表，便于筛 `is_retry`（同步更新） |
| `token_usage_by_comment.jsonl` | 每完成一条 append 一行：该 comment 切片 + 累计 `billing_so_far` |

每条请求含 `attempt` / `is_retry` / `exclude_from_clean_total`；汇总见 `billing.exclude_retries`（去重试虚高）与 `billing.retry_only`。

## 运行

```bash
conda activate verl
cd /path/to/llm-anonymization-causal-input-tokens

# 建议 screen；按机器改 CUDA_VISIBLE_DEVICES 与模型路径
export CAUSAL_VLLM_MODEL_PATH=/path/to/Llama-3.1-8B-Instruct
CUDA_VISIBLE_DEVICES=2,3 python zxz_run_causal_vllm_token_stats.py \
  --output-dir results/vllm_llama3.1-8B_all300_tokens \
  --model-path "$CAUSAL_VLLM_MODEL_PATH" \
  --model-name "$CAUSAL_VLLM_MODEL_PATH" \
  --vllm-port 8010 \
  --profile-workers 8 \
  --max-model-len 32768 \
  --max-output-tokens 8192

# 冒烟 1 人
CUDA_VISIBLE_DEVICES=0 python zxz_run_causal_vllm_token_stats.py \
  --limit-profiles 1 --profile-workers 1

# 只校验因果调度（不启模型）
python zxz_run_causal_vllm_token_stats.py --dry-run --limit-profiles 1
```

已有 `pers*/result.json` 会跳过（可断点续跑）。手动汇总：

```bash
python zxz_aggregate_token_usage.py \
  --output-dir results/vllm_llama3.1-8B_all300_tokens
```

## 文件说明

| 文件 | 作用 |
|---|---|
| `zxz_run_causal_vllm_token_stats.py` | **主入口**：包装 vendored v2 因果脚本 + 默认 300 人 / vLLM |
| `zxz_aggregate_token_usage.py` | 汇总各 pers 的 `token_usage.json` |
| `vendor/llm-anonymization-v2/` | 自包含驱动与 `src/`（勿依赖机外 checkout） |
| `data/inputs/all300_authors.txt` | 300 个 `pers*` 列表 |
| `zxz_estimate_offline_stub.py` | 旧离线 stub 估算（不调模型；anon prompt 会低估） |
| `results/all300_causal_full/` | 旧 stub 估算结果（保留作对照） |

## 注意

- 真实跑数很重：约 7823 条 comments × 每条约 9 次 LLM 调用（3×infer+anon+utility）。
- Token 以 vLLM 返回的 `usage` 为准（`usage_source: api`）。
- 改端口/显存：`--vllm-port`、`--gpu-memory-utilization`、`--max-model-len`。
- **mismatch 默认不重试**：匿名输出条数对不上时，立刻用当前 comment 原文作为匿名结果继续（仍计这次 anon 的 token）。加 `--strict-align-retry` 可恢复 v2 原版重试。
- 目标机需已安装 `vllm` / `transformers` 等运行时依赖；代码与 SynthPAI 数据已在本目录内。
