# coding-agent 内部架构

本文档描述本 agent 自身的设计，供扩展 / 二次开发参考。

## 1. 运行流

```
用户输入
  │
  ▼
cli.py ──► run_agent(cfg, task, mode)
  │          │ 1. build_graph(): 构建 StateGraph
  │          │ 2. 组装 messages: [System(prompt), Human(task)]
  │          │ 3. graph.stream(..., stream_mode="values")   ← 流式打印进度
  │          ▼
  └── 最终 state（含 final_summary / messages）
```

三种模式只改变 **System Prompt** 和 **任务文本**：

| mode | System Prompt | 结束方式 |
|------|---------------|----------|
| `analyze` | 要求输出 `<final_analysis>...</final_analysis>` 报告 | 模型调 `finish` |
| `run` | 要求改代码并自验证 | 模型调 `finish` |
| `chat` | 自由问答，可调用工具 | 模型直接输出文字 |

## 2. LangGraph 图

节点：

- **agent**：`llm.bind_tools(tools)` 调用模型；若历史超预算先 `trim_messages`。
- **tools**：`ToolNode` 顺序执行模型请求的所有工具。
- **finalize**：扫描消息里 `finish` 工具调用的 `args.summary`，写入 `final_summary`。

边（conditional routing）：

- `agent → tools`：最后一条消息是带 `tool_calls` 的 `AIMessage`。
- `agent → finalize`：模型直接输出文字（无工具调用）。
- `tools → agent`：最后一条 `ToolMessage` 不是 `finish`。
- `tools → finalize`：最后一条 `ToolMessage` 是 `finish`（结束循环）。

`recursion_limit = max_iterations + 10` 兜底，防止无限循环。

## 3. 工具层如何拿到配置

工具不直接 import `Config`（避免循环依赖），而是通过 LangChain 的
`RunnableConfig` 注入机制读取 invoke 时放进 `configurable` 的值：

```python
@tool
def read_file(path: str, ..., config: RunnableConfig = None) -> str:
    root = Path((config or {}).get("configurable", {}).get("project_root", "."))
```

> 注意：`config` 参数必须注解为裸 `RunnableConfig` 类型，LangChain 才会把它当作
> 注入参数（从工具 schema 中排除并在运行时注入）；写 `Optional[RunnableConfig]`
> 会被误当成模型需要填的参数。

`cli._invoke_config()` 负责把 `project_root / read_only / 各类 limit / timeout`
塞进 `configurable`。这样同一个工具函数在不同 root 下可复用。

## 4. 安全设计

- 所有文件操作前做 `_within(root, p)` 校验，禁止逃出工程根目录。
- `read_only=True` 时 `build_tools()` 直接从工具列表移除
  `write_file / edit_file / delete_file / run_shell`（模型根本拿不到这些工具）。
- `run_shell` 有 `timeout` 和输出长度上限，避免挂死和刷屏。

## 5. 上下文管理

按字符数粗估 token（`_char_counter`），超过 `context_budget_chars` 时用
`trim_messages(strategy="last")` 丢弃最旧的（保留 system）。当前是"丢弃式"，
没有跨轮次摘要 —— 这是后续可增强的点（插入一个 summarize 节点用 LLM 压缩历史）。

## 6. 扩展点

- **新增模式**：在 `prompts.py` 加一段 prompt + `cli.py` 加子命令即可。
- **新增工具**：在 `tools/` 新增 `@tool` 函数并注册到 `ALL_TOOLS`。
- **接入 VS Code LSP**：可加一个 `get_diagnostics` 工具调用语言服务器，或复用
  `run_shell` 跑 `tsc --noEmit` / `pytest` 等。
- **加 checkpointer**：把 `graph.compile(checkpointer=...)` 接上
  `MemorySaver` / `SqliteSaver`，就能在 `chat` 模式实现真正的持久会话与
  `get_state` / `update_state`。
