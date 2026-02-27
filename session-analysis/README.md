# Session Analysis

Tools and documentation for analyzing AI coding assistant session data from Claude Code and Codex CLI.

## Why Session Analysis?

When using AI coding assistants for non-trivial work (like our FSR4 kernel optimization campaign), it's useful to understand:

- **How long things actually took** - wall time vs active compute time
- **How much AI capacity was used** - token counts, turn counts, tool calls
- **What the workflow looked like** - interactive vs autonomous phases, idle gaps
- **How the human-AI collaboration unfolded** - user message timeline, session phases

This helps with reproducibility planning, cost estimation, and understanding the practical dynamics of AI-assisted development.

## Session Data Locations

### Claude Code

Sessions are stored as JSONL files under:
```
~/.claude/projects/<encoded-project-path>/<session-uuid>.jsonl
```

The project path is the filesystem path with `/` replaced by `-`, e.g.:
```
~/.claude/projects/-home-lhl-github-lhl-fsr4-rdna3/af1c788b-c174-4d2f-bf69-56e2d5ebda85.jsonl
```

Subagent sessions live in a `subagents/` subdirectory within the project.

### Codex CLI

Sessions are stored under:
```
~/.codex/sessions/YYYY/MM/DD/rollout-<ISO-timestamp>-<uuid>.jsonl
```

Additional data:
- `~/.codex/history.jsonl` - consolidated history across all sessions
- `~/.codex/state_5.sqlite` - session state database
- `~/.codex/config.toml` - CLI configuration

## JSONL Format Reference

### Claude Code Entry Types

| Type | Description | Key Fields |
|------|-------------|------------|
| `user` | User message | `message.content` (text), `timestamp` |
| `assistant` | Assistant response | `message.usage` (token counts), `message.model`, `message.content` |
| `progress` | Tool execution progress | `data` (progress info), `toolUseID` |
| `system` | System events (hooks, etc.) | `slug`, `subtype`, `hookInfos` |
| `file-history-snapshot` | File state snapshots | `snapshot` (file contents) |

**Token usage** (in `assistant` entries under `message.usage`):
- `input_tokens` - direct input tokens
- `output_tokens` - generated output tokens
- `cache_creation_input_tokens` - tokens used to create prompt cache
- `cache_read_input_tokens` - tokens served from prompt cache

### Codex CLI Entry Types

| Type | Description | Key Fields |
|------|-------------|------------|
| `session_meta` | Session metadata | `payload.cwd`, `payload.id`, `payload.cli_version`, `payload.model` |
| `turn_context` | Per-turn context | `payload.model`, `payload.approval_policy`, `payload.turn_id` |
| `event_msg` | Lifecycle events | `payload.type` (see subtypes below) |
| `response_item` | Messages & tool calls | `payload.role`, `payload.content`, `payload.type` |
| `compacted` | Compacted history | `payload.replacement_history` (array of prior messages) |

**Event message subtypes** (`event_msg.payload.type`):
- `task_started` / `task_completed` - turn lifecycle
- `model_response_started` / `model_response_completed` - inference lifecycle
- `token_count` - running token usage (has `info.total_token_usage`)
- `exec_started` / `exec_completed` - tool execution lifecycle
- `exec_approval_pending` / `exec_approval_granted` - approval flow
- `reasoning` - model reasoning events

**Token usage** (in `token_count` events under `info.total_token_usage`):
- `input_tokens` - total input tokens (cumulative)
- `cached_input_tokens` - tokens served from cache
- `output_tokens` - total output tokens
- `reasoning_output_tokens` - tokens used for chain-of-thought reasoning
- `total_tokens` - input + output total

## Usage

### Basic: All sessions for a project

```bash
python3 analyze_sessions.py --project-filter fsr4
```

### JSON output (for downstream processing)

```bash
python3 analyze_sessions.py --project-filter fsr4 --json > sessions.json
```

### Custom idle threshold

By default, gaps > 5 minutes between events are considered idle. To use 10 minutes:

```bash
python3 analyze_sessions.py --idle-threshold 600
```

### Custom data directories

```bash
python3 analyze_sessions.py \
  --claude-dir /path/to/.claude/projects \
  --codex-dir /path/to/.codex \
  --project-filter myproject
```

### Pipe JSON to jq for specific queries

```bash
# Get just the big Codex session's token usage
python3 analyze_sessions.py --project-filter fsr4 --json | \
  jq '.[] | select(.tool == "codex-cli" and .wall_seconds > 3600) | .tokens'

# List all user messages from all sessions
python3 analyze_sessions.py --project-filter fsr4 --json | \
  jq '.[] | select(.user_messages) | .user_messages[] | .text'

# Get wall time and active time for all sessions
python3 analyze_sessions.py --project-filter fsr4 --json | \
  jq '.[] | {session_id: .session_id[:12], tool, wall: .wall_seconds, active: .active_seconds}'
```

## Extending for Future Projects

The script auto-discovers sessions by scanning the standard data directories. To analyze sessions from a different project, just change `--project-filter`:

```bash
python3 analyze_sessions.py --project-filter shisad
python3 analyze_sessions.py --project-filter mistral-vibecheck
```

To add support for another AI coding tool, implement an `analyze_<tool>_session()` function that returns a dict with at minimum:
- `tool`, `session_id`, `file`
- `start`, `end`, `wall_seconds`, `active_seconds`, `idle_seconds`
- `tokens` dict with whatever breakdown the tool provides

## Notes on Data Interpretation

- **Claude Code cache tokens** dominate total API token counts. The `cache_read_input_tokens` are served from Anthropic's prompt cache and are much cheaper than regular input tokens. A session showing "4M total API tokens" may have only 6K regular input + 13K output, with the rest being cache operations.

- **Codex CLI token counts are cumulative** - the `token_count` events report running totals, so the script takes the last value as the session total.

- **Compacted history creates duplicate user messages** - when Codex compacts the conversation context, it replays earlier messages. The script deduplicates by matching the first 80 characters of each message.

- **Wall time vs active time** - idle gaps often represent the human stepping away, not the AI waiting. In autonomous sessions, the AI may finish quickly and the "idle" time is just the session sitting open until the human returns.

- **Turn context count** in Codex represents the number of model invocations (inference calls), which is higher than user turns because the model makes multiple calls per user message when using tools.
