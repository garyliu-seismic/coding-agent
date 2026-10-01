# coding-agent 内部架构

本文档描述 agent 自身的设计，供扩展 / 二次开发参考。

---

## 1. 运行流

```
用户输入
  │
  ▼
cli.py ──► run_agent(cfg, task, mode)
             │ 1. build_graph(): 构建 StateGraph
             │ 2. 组装 messages: [System(prompt), Human(task)]
             │ 3. graph.stream(..., stream_mode="values")  ← 流式打印进度
             ▼
           最终 state（含 final_summary / messages）
```

三种模式只改变 **System Prompt** 和任务文本：

| mode | System Prompt | 结束方式 |
|------|---------------|----------|
| `analyze` | 要求输出架构报告 | 模型调 `finish` |
| `run` | 要求改代码并自验证 | 模型调 `finish` |
| `chat` | 自由问答，可调用工具 | 模型直接输出文字 |

---

## 2. LangGraph 图

```
START → agent ──► tools ──► agent (循环)
                      └──► finalize → END
        agent ──► finalize → END  (无工具调用时)
```

节点：

- **agent**：`llm.bind_tools(tools)` 调用模型；超预算先做 compact 摘要压缩。
  根据状态选择三档模型：
  - `model_full`：完整工具集。
  - `model_no_explore`：检测到探索性循环时，移除 list/search 工具。
  - `model_only_finish`：超时或长期停滞时，只保留 `finish`。
- **tools**：`ToolNode` 顺序执行工具调用；视觉工具有额外后处理（见第 5 节）。
- **finalize**：扫描消息里 `finish` 的 `args.summary`，写入 `final_summary`。

路由逻辑：

```python
# agent → ?
last.tool_calls  → "tools"
otherwise        → "finalize"

# tools → ?
last_tool.name == "finish"  → "finalize"
otherwise                   → "agent"
```

`recursion_limit = max_iterations * 2 + 20`（每个 agent↔tools 往返消耗 2 个节点）。

---

## 3. 工具层如何拿到配置

工具不直接 import `Config`，而是通过 LangChain `RunnableConfig` 注入机制：

```python
@tool
def read_file(path: str, config: RunnableConfig = None) -> str:
    root = Path((config or {}).get("configurable", {}).get("project_root", "."))
```

> `config` 必须注解为裸 `RunnableConfig`（不能写 `Optional[RunnableConfig]`），
> LangChain 才会将它从工具 schema 中排除并在运行时自动注入。

`cli._invoke_config()` 将以下字段塞进 `configurable`：

| 字段 | 来源 | 用途 |
|------|------|------|
| `project_root` | `cfg.project_root` | 文件路径安全校验基准 |
| `read_only` | `cfg.read_only` | 工具层二次拦截 |
| `allow_shell` | `cfg.allow_shell` | Shell 工具开关 |
| `shell_timeout` | `cfg.shell_timeout` | Shell 超时 |
| `tool_output_limit` | `cfg.tool_output_limit` | Shell 输出长度上限 |
| `file_read_limit` | `cfg.file_read_limit` | 文件读取长度上限 |
| `vision_model_url` | `cfg.vision_model_url` | 视觉模型 endpoint |
| `vision_model_name` | `cfg.vision_model_name` | 视觉模型 id（默认 `gpt-5-mini`）|
| `vision_model_api_key` | `cfg.vision_model_api_key` | 视觉模型 API key |
| `vision_timeout` | `cfg.vision_timeout` | 视觉请求超时 |
| `vision_max_retries` | `cfg.vision_max_retries` | 视觉请求重试次数 |

---

## 4. 安全设计

- **路径越界校验**：所有文件操作前执行 `_within(root, p)`，禁止逃出工程根目录。
- **只读模式**：`build_tools()` 直接移除 `write_file / edit_file / delete_file / run_shell / move_file`，模型根本拿不到这些工具。
- **Shell 限制**：有 `timeout` 和输出长度上限（`tool_output_limit`），避免挂死和刷屏。
- **视觉路径安全**：`describe_image` / `view_image` 同样复用 `_within(root, p)` 校验。

---

## 5. 视觉（Image Recognition）架构

视觉功能由两个独立子系统组成，解耦于主 LLM：

### 5.1 Inline 视觉：`view_image`

```
image file
  → view_image (Pillow resize + base64 encode)
  → 返回 {"type":"image_url","url":"data:image/png;base64,..."}
  → graph._route_after_tools() 把它注入为 HumanMessage(IMAGE_CONTENT_JSON)
  → 主 LLM 接收像素数据，直接回答
```

用于让**主 LLM** 直接看图（需要主模型本身支持多模态）。

### 5.2 描述视觉：`describe_image`

```
image file
  → _resize_to_bytes()  (Pillow，max_side=1568，max_bytes=256KB)
  → base64 encode
  → _build_request_body() → OpenAI Chat Completions JSON
  → _post_with_retry()   → POST /v1/chat/completions
  → _extract_text()      → 纯文本描述
  → 返回文字给主 LLM
```

**独立视觉模型**，默认 `gpt-5-mini`（Azure AI Foundry）。
主 LLM 收到的是**文字描述**，不涉及像素，适合主模型不支持多模态的场景。

#### 5.2.1 依赖

- **零额外依赖**：仅用 Python 标准库（`urllib.request`, `base64`, `json`, `io`）。
- **Pillow**：已由 `image_view.py` 引入，复用。

#### 5.2.2 后端兼容性

两个后端使用**相同的 JSON 请求格式**（OpenAI Chat Completions），仅以下字段不同：

| 参数 | Ollama 本地 | gpt-5-mini (Azure/OpenAI) |
|------|-------------|--------------------------|
| endpoint | `http://localhost:11434/v1/chat/completions` | `https://<resource>.services.ai.azure.com/openai/v1/chat/completions` |
| model | `gemma4:12b` / `qwen3.8:latest` | `gpt-5-mini` |
| token 限制字段 | `max_tokens` | `max_completion_tokens` |

> 代码按模型名前缀自动切换：`gpt-5-*` / `o1-*` / `o3-*` / `o4-*` → `max_completion_tokens`，其余 → `max_tokens`。

#### 5.2.3 重试策略

```python
_post_with_retry(url, body, headers, timeout, max_retries):
  for attempt in range(max_retries + 1):
    try: POST ...
    except HTTP 429: sleep(Retry-After or 2^attempt); retry
    except HTTP 5xx: sleep(2^attempt); retry
    except connect error: sleep(2^attempt); retry
  raise RuntimeError(...)
```

#### 5.2.4 配置优先级

```
RunnableConfig.configurable.vision_model_url
  > env CODING_AGENT_VISION_MODEL_URL
  > env DEEPSEEK_BASE_URL
  > "http://localhost:11434/v1"  (fallback)
```

模型名和 API key 同理。

### 5.3 视觉工具注册条件

```python
if cfg.vision != "off":
    tools += [view_image, describe_image, get_omitted_image]
```

`read_image_meta` 始终注册（不涉及像素，安全）。

---

## 6. 上下文管理（Compaction）

```
count_chars(messages) > context_budget_chars (120k)?
  Yes → compact(messages, llm_plain, total_budget, keep_recent)
          │  1. 用无工具绑定的 LLM 生成历史摘要 SystemMessage
          │  2. 保留最近 keep_recent_chars 字符的消息
          │  3. 保留最近 keep_recent_images 张图片，旧图替换为占位符
          └→ 返回压缩后的 messages
```

使用 **LLM 摘要**而非简单丢弃，避免切断 AIMessage/ToolMessage 配对导致 API 报错。
图片单独计费（`image_token_cost`，默认 2048 字符当量）以防 VRAM 溢出。

---

## 7. 循环保护（Loop Guard）

`_loop_warning()` 检测两类卡死模式：

1. **重复调用**：同一工具 + 同一参数在近 20 次内出现 ≥3 次 → 注入警告 + 移除探索工具。
2. **探索停滞**：连续 6 次工具调用全是 `list_directory / file_search / grep_search` → 同上。

`_stall_count()` 检测连续无进展（无写/编辑/Shell/finish）步数，超过阈值（12步）→ 切换 `model_only_finish`。

---

## 8. 扩展点

| 扩展方向 | 实现方式 |
|---------|---------|
| 新增模式 | `prompts.py` 加 prompt + `cli.py` 加子命令 |
| 新增工具 | `tools/` 下写 `@tool` + 注册到 `ALL_TOOLS` |
| 换视觉后端 | 设 `CODING_AGENT_VISION_MODEL_*` 环境变量即可，无需改代码 |
| 持久会话 | `graph.compile(checkpointer=SqliteSaver(...))` |
| VS Code LSP | 加 `get_diagnostics` 工具调用语言服务器 |
| 执行沙箱 | 替换 `run_shell` 为 Docker exec，传入容器 ID |
