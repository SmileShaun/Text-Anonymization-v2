你是一名资深工程师。当前工作区（即/hdd/zxz/project/Text-Anonymization/baseline/IntentAnony）是某个**文本匿名化 baseline 仓库**：其原始实现通常会把一个 profile（pers）的**全部 comments 拼接后**做匿名化（非因果、全文可见）。

请在该仓库中新增一个**因果匿名化执行脚本**。要求：

- **必须**复用该仓库已有的匿名化能力（推理、改写、解析、对齐等），**不得重写**其匿名化模型/提示词本身
- 只新增一层**因果调度**：控制每一步可见哪些评论、如何定稿、如何并行
- **所有新写的 Python 文件必须以 `zxz_` 开头**（含主脚本、辅助模块、工具脚本等；不得新增非 `zxz_` 前缀的 `.py` 文件）
- **不保留原仓库对 comments 条数的截断**：若原实现会限制每个 profile 只处理前 K 条（或类似 `max_comments` / 采样 / 截断），因果版**必须关闭或绕过该限制**，对输入文件中的**全部 comments** 做匿名化（`--limit-comments` 仅用于调试/试跑，正式跑全量时不要沿用原仓库默认截断）
- 本 Prompt 会用于多个不同 baseline 仓库；请先阅读当前仓库的结构与入口，再按其接口适配，不要假设某个固定文件名或固定类名
- 多轮 / 单轮行为必须**跟随原仓库**：原仓库是多轮迭代才做多轮；原仓库只有一轮，则因果版也只做一轮，不要自行加多轮

---

## 一、因果算法要求（硬性）

对每个 profile，按评论下标 `M = 0..N-1` **串行**处理：

1. **可见范围**  
   处理第 `M` 条时，模型只能看到：
   - 已定稿的匿名前缀 `fixed_anon[0..M-1]`
   - 当前第 `M` 条（首轮为原文；若原仓库支持多轮，之后为当前轮改写结果）  
   **禁止**看到未来评论 `M+1..N-1`。

2. **每条评论的迭代轮数与步骤：跟随原仓库**  
   - 若原仓库对全文匿名是**多轮** refinement：因果版在每条评论上也做同样轮数、同样步骤顺序（例如是否含 infer / anonymize / utility，以及 utility 是否参与后续决策），**不要自行增删步骤或改变其作用**。  
   - 若原仓库只有**一轮**：因果版对每条评论也只跑一轮，不要额外发明多轮选项。

3. **历史冻结 / 只定稿当前条（最重要）**  
   - 模型输出可能改写整段可见前缀，但提交时必须**强制** `0..M-1` 仍等于已定稿 `fixed_anon`  
   - **只采纳第 `M` 条**的新文本（“落子”）  
   - 跨评论推进时：只有第 `M` 条落子后才 `fixed_anon.append(...)`，再进入 `M+1`  
   - **仅当原仓库为多轮时**，还必须保证**轮内**历史冻结，说明如下：  
     - 处理第 `M` 条时，某一轮 anonymize 的模型输出**有可能**顺带改写前面的 `0..M-1`（不保证只改当前条）  
     - 一旦出现这种情况，**禁止**把这些“被改写过的历史版本”当作下一轮的输入前缀  
     - 下一轮（以及之后每一轮）构造给模型的上下文时，Prompt Template 里的前缀**必须始终是已经落子的 comments**：即 `fixed_anon[0..M-1]`（已定稿、不可变），再加上**当前条**上一轮落子后的文本  
     - 换句话说：历史前缀以落子结果为准；若模型改动了 `0..M-1`，这些改动一律丢弃，不得进入下一轮 Prompt

4. **失败回退**  
   某次调用失败时：保留上一轮成功文本（若无则回退原文），记录 error/status，便于续跑与排查；不要静默丢数据。

5. **自检**  
   dry-run 或日志中应能证明：任意 `M` 的可见前缀长度均为 `M+1`，且不含未来评论；并有强制逻辑/断言保证只定稿当前条（多轮仓库还需断言轮内历史冻结）。

---

## 二、工程能力要求

脚本需同时支持：

### 1. API 模式
- OpenAI-compatible HTTP API（`--base-url` / `--api-key` / `--model-name`）

### 2. vLLM 模式
- 可拉起/关闭本地 vLLM server（`--model-path`、host/port、gpu-memory-utilization、max-model-len 等）
- **必须默认关闭思考/reasoning 模式**  
  （例如请求里传 `chat_template_kwargs.enable_thinking=false`；提供 `--disable-thinking` / `--enable-thinking`，但**默认关闭**）

### 3. 并行
- 实现 `--profile-workers`：多个 profile 并行  
- **单个 profile 内评论必须串行**（因果约束）

### 4. 其他
- retries、dry-run、log-level  
- 原子写入 `result.json`  
- 若目标文件已存在可跳过（断点续跑）  
- **必须**统计并写出每个 pers 的输入/输出 token（见下一节）

若当前仓库已有 API/vLLM 客户端或 server 封装，优先复用并改造成满足上述因果调度；没有则按仓库风格新增轻量封装（新增封装文件同样必须以 `zxz_` 开头）。

---

## 三、数据与 I/O（本环境默认路径）

- 默认 profiles 目录：`/hdd/zxz/project/Text-Anonymization/data/synthpai/profiles`（`pers*.json`）
- 必须支持 `--profile-list`，例如：  
  `/hdd/zxz/project/Text-Anonymization/data/synthpai/top30_most_comments.txt`  
  （每行一个 author，如 `pers33`）
- 输出：`--output-dir/<author>/result.json`  
  建议字段：`author`、`comments[]`（含 `index/original/anonymized/status`，多轮仓库可含 `rounds`）、模型与轮次元数据  
  **必须包含该 pers 的 token 用量统计**，至少包括：
  - 输入 token 总量（prompt / input）
  - 输出 token 总量（completion / output）
  - 可选：按调用类型拆分（infer / anonymize / utility 等）、调用次数  
  以 API/vLLM 返回的 usage 为准；无法取得时需明确记录并尽量给出可复现的估算说明

不同仓库的 profile JSON 字段可能不同：请先探测当前仓库/数据格式，再做适配（评论文本、用户名、GT/PII 标签等）。

---

## 四、CLI 最低要求

```text
--baseline-repo                 # 若需要把当前/指定仓库加入 PYTHONPATH
--profiles-dir                  # default: .../data/synthpai/profiles
--profile-list                  # optional, e.g. top30_most_comments.txt
--output-dir                    # required
--backend {api,vllm}
--base-url --api-key --model-name
--model-path --vllm-host --vllm-port --vllm-startup-timeout
--gpu-memory-utilization --max-model-len --max-output-tokens
--disable-thinking / --enable-thinking   # 默认关闭 thinking
--temperature --top-k/--top-p --request-timeout
--profile-workers
--retries
--max-refinement-rounds         # 仅当原仓库本身支持多轮时提供/生效；单轮仓库不要强行加入
--limit-profiles --limit-comments   # 仅调试用；勿把原仓库默认 comments 截断带到正式全量跑
--dry-run --log-level
```

脚本**必须**以 `zxz_` 开头，并建议命名体现 **causal + frozen history**（例如 `zxz_causal_frozen_anonymize.py`）；若拆出辅助 `.py` 模块，也一律 `zxz_*.py`。放在该 baseline 仓库的合适位置。实现后，文件头与 README/说明中必须给出与下列等价的可复制运行命令（脚本路径按实际落盘位置替换）。

说明：若原仓库加载 profile 时会截断 comments（例如只取前 N 条），因果脚本加载数据时**必须读取并处理全部 comments**，不得继承该默认截断。

### 示例运行指令 A：API（deepseek-chat）

```bash
python /path/to/baseline/zxz_causal_frozen_anonymize.py \
  --backend api \
  --baseline-repo /path/to/baseline \
  --profiles-dir /hdd/zxz/project/Text-Anonymization/data/synthpai/profiles \
  --profile-list /hdd/zxz/project/Text-Anonymization/data/synthpai/top30_most_comments.txt \
  --output-dir /hdd/zxz/project/Text-Anonymization/baseline/<baseline_name>/result/causal_frozen_a_deepseek-chat_i_deepseek-chat \
  --base-url https://api.deepseek.com/v1 \
  --api-key "${DEEPSEEK_API_KEY}" \
  --model-name deepseek-chat \
  --temperature 0.1 \
  --top-k 0.9 \
  --request-timeout 300 \
  --disable-thinking \
  --profile-workers 16 \
  --retries 3 \
  --max-refinement-rounds 3 \
  --log-level INFO
```

说明：`--max-refinement-rounds` 仅当该 baseline 原仓库本身支持多轮时生效；单轮仓库可省略。`--api-key` 也可用环境变量注入，勿把真实 key 写进仓库。

### 示例运行指令 B：vLLM（Llama-3.1-8B-Instruct）

```bash
CUDA_VISIBLE_DEVICES=0 python /path/to/baseline/zxz_causal_frozen_anonymize.py \
  --backend vllm \
  --baseline-repo /path/to/baseline \
  --profiles-dir /hdd/zxz/project/Text-Anonymization/data/synthpai/profiles \
  --profile-list /hdd/zxz/project/Text-Anonymization/data/synthpai/top30_most_comments.txt \
  --output-dir /hdd/zxz/project/Text-Anonymization/baseline/<baseline_name>/result/causal_frozen_a_Llama-3.1-8B-Instruct_i_Llama-3.1-8B-Instruct \
  --model-path /home/zxz/ckpt/LLama3/Llama-3.1-8B-Instruct \
  --model-name /home/zxz/ckpt/LLama3/Llama-3.1-8B-Instruct \
  --vllm-host 127.0.0.1 \
  --vllm-port 8000 \
  --vllm-startup-timeout 3600 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 32768 \
  --max-output-tokens 8192 \
  --temperature 0.0 \
  --top-k 0.9 \
  --request-timeout 600 \
  --disable-thinking \
  --profile-workers 8 \
  --retries 3 \
  --max-refinement-rounds 3 \
  --log-level INFO
```

说明：vLLM **必须**关闭思考模式（`--disable-thinking`，且默认即为关闭）。`--max-model-len` / `--max-output-tokens` 可按机器显存与该模型上下文上限调整；多卡时设置相应 `CUDA_VISIBLE_DEVICES`。

### Dry-run（任选后端，先验证因果前缀）

```bash
python /path/to/baseline/zxz_causal_frozen_anonymize.py \
  --backend api \
  --profiles-dir /hdd/zxz/project/Text-Anonymization/data/synthpai/profiles \
  --profile-list /hdd/zxz/project/Text-Anonymization/data/synthpai/top30_most_comments.txt \
  --output-dir /tmp/causal_frozen_dry_run \
  --base-url https://api.deepseek.com/v1 \
  --api-key "${DEEPSEEK_API_KEY}" \
  --model-name deepseek-chat \
  --disable-thinking \
  --limit-profiles 1 \
  --dry-run \
  --log-level INFO
```

---

## 五、验收标准

1. **因果正确**：无未来泄漏；只定稿当前条；若原仓库多轮，则轮内历史冻结。  
2. **迭代策略正确**：多轮/单轮严格跟随原仓库，不擅自加轮。  
3. **必须复用**原仓库匿名化能力，不重写模型/提示词。  
4. **后端可用**：API 与 vLLM 两条路径都能跑；思考模式默认关闭。  
5. **并行正确**：`--profile-workers > 1` 并行多个 pers；同一 pers 评论串行。  
6. **数据可选子集**：支持 profiles 目录 + `top30_most_comments.txt`。  
7. **Token 统计**：每个 pers 的 `result.json` 含输入/输出 token 统计。  
8. **文档**：文件头给出与上文等价的 API（deepseek-chat）与 vLLM（Llama-3.1-8B-Instruct）完整运行命令。  
9. **改动范围**：以新增脚本为主；不重构仓库原有匿名化核心；不要引入“轮内可改写已定稿历史”的错误语义（适用于多轮仓库）。  
10. **命名约束**：本次任务新增的所有 Python 文件均以 `zxz_` 开头；不得新增其他前缀的 `.py` 文件。  
11. **全量 comments**：不沿用原仓库对 comments 条数的截断；正式跑时对每个输入 profile 的全部评论做因果匿名化（仅 `--limit-comments` 可显式截断用于调试）。

---

## 六、交付

1. 实现脚本（及必要辅助模块，文件名均以 `zxz_` 开头）  
2. 简短说明：如何 dry-run、如何用上述 API / vLLM 命令正式跑 top30  

先阅读**当前** baseline 仓库的代码结构、数据加载方式与模型调用入口，再实现；不要依赖某个外部仓库的具体文件路径或符号名称。
