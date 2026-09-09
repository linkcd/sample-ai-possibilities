# Implementation Instructions: Bound Conversation History in the Combined Gateway+Memory Agent

## Audience
This is a task brief for a coding assistant (human or AI) implementing changes in this
repo. It contains full background so you don't need the original investigation
conversation to understand *why* this change is needed.

## Background — how we got here

### The ask
The team is building a new variant of the AI soccer agents that combines:
- **Gateway tools** (from `ai-team-strands-gateway/`) — MCP tools served via AgentCore
  Gateway (`find_open_space`, `evaluate_shot`, `calculate_pass_options`,
  `get_defensive_assignment`) that let the agent call out to Lambda-computed tactical
  analysis instead of reasoning about raw coordinates itself.
- **Memory** (from `ai-team-strands-memory/`) — `AgentCoreMemorySessionManager`, which
  persists every conversation turn to an AgentCore Memory resource and reloads it on
  each invocation, so agents can recall things across ticks (e.g., "which opponent shoots
  most often").

The question was whether these two additions help or hurt the runtime latency problem
already observed in the baseline (`ai-team-strands-balanced/`) deployment.

### The root cause we found (baseline investigation)
Using AgentCore CloudWatch Logs (OTel trace spans in the `spans` log stream, e.g. under
`/aws/bedrock-agentcore/runtimes/ai_gk_agent-*-DEFAULT`), we found:

- Each position agent (`ai-gk`, `ai-def`, `ai-mid`, `ai-fwd1`, `ai-fwd2`) creates **one
  Strands `Agent` object at cold start** and reuses it for every subsequent tick
  (`agent = create_agent(SYSTEM_PROMPT, model_id=...)` at module scope in each
  `src/main.py`).
- Strands' default `Agent` behavior appends every `agent(state_summary)` call's
  user message and the model's assistant response to an **in-memory, unbounded**
  message history. On the next tick, the *entire accumulated history* is resent to
  Bedrock as part of the prompt — not just the current tick's game state.
- Confirmed directly from a real trace span (`ai_gk_agent`, Nova Micro):
  - A single per-tick `chat` span (one Bedrock Converse call) had
    `gen_ai.usage.input_tokens: 8282` — already large for what should be a ~500-token
    system prompt + ~150-token state block.
  - The parent `invoke_agent` span covering the same tick showed
    `gen_ai.usage.prompt_tokens: 568509` — over half a million tokens sent in that one
    agent invocation, because the event loop was silently carrying dozens of prior
    ticks' user/assistant message pairs (visible as a long chain of
    `gen_ai.user.message` / `gen_ai.assistant.message` events in the span, going back
    many game-clock seconds).
  - `summarize_state()` (in `lib/state.py`) itself is small and well-designed — a
    single game tick's text summary is roughly 150–250 tokens. **It is not the cost
    driver.** The driver is unbounded conversation-history replay.
- CloudWatch aggregate stats (via Logs Insights across all 5 balanced agents) showed
  average latency 700–1,200ms per Bedrock call, with the higher-latency/higher-token
  agents correlating with larger accumulated history, not model choice.

### Why switching models (e.g., to Claude Haiku) does not fix this
We evaluated switching from Nova Micro/Lite/Pro to Claude Haiku. Public benchmarks and
AWS's own latency-optimized-inference docs show Haiku is not reliably faster than Nova
Micro for short-context tasks, and can be slower. More importantly: **the token bloat is
architectural** (unbounded history), not a property of which model processes the tokens.
Any model, including Haiku, would face the same growing-prompt problem if history keeps
accumulating.

### How gateway tools and memory interact with this root cause
We reviewed the actual code in both folders:

- **`ai-team-strands-gateway/gateway_agent_base.py`**: builds a **stateless**
  `Agent(model=model, system_prompt=system_prompt, tools=tools)` — no session manager,
  no conversation persistence. Gateway tools add tool-schema tokens to every request and
  can add extra round-trips when the model chooses to invoke a tool, but they do **not**
  touch the conversation-history problem. On their own, gateway tools are safe from a
  history-growth perspective.

- **`ai-team-strands-memory/memory_agent_base.py`**: builds
  `Agent(model=model, system_prompt=system_prompt, session_manager=session_manager)`
  where `session_manager = AgentCoreMemorySessionManager(AgentCoreMemoryConfig(memory_id=...,
  session_id=f"match-{team_id}-{position_label}", actor_id=f"{team_id}-{position_label}"))`.
  **No window/truncation parameter is set.** This session manager persists every turn to
  AgentCore Memory and reloads the full history on each invocation. This **relocates**
  the unbounded-history problem from in-process memory to a durable external store — it
  does not bound it. If anything, this is *worse* than the baseline, because:
  1. History now survives container restarts (in the baseline, a fresh container at
     least resets the runaway context).
  2. There is no `batch_size`, window, or truncation config applied, so growth is
     unbounded for the life of the AgentCore Memory session.

### Conclusion driving this task
When combining gateway tools + memory into one new agent variant, we must **explicitly
bound the conversation window** sent to the model on every call. Without this, the
combined agent inherits the worst of both: unbounded history (from memory) *plus*
tool-schema token overhead (from gateway) on every single call, compounding the existing
latency/cost problem rather than fixing it.

The fix must preserve the two things gateway+memory are actually good for:
- Gateway tools' tactical computation (keep as-is).
- Memory's actual value-add, which is **long-term recall of a few durable facts**
  (e.g., "opponent player 3 shoots frequently from range"), not verbatim replay of every
  tick's raw coordinates.

## The fix

Strands' `conversation_manager` is a separate, independent constructor argument on
`Agent` from `session_manager`. It operates on the in-memory message list *regardless of
where those messages came from* (freshly generated this session, or reloaded from
AgentCore Memory by the session manager). This means we can keep `session_manager` (for
durable cross-session memory / long-term strategies) while adding a
`conversation_manager` that caps how many of those messages are actually resent to the
model on each call.

Use `SlidingWindowConversationManager` (Python import:
`from strands.agent.conversation_manager import SlidingWindowConversationManager`):

```python
class SlidingWindowConversationManager(ConversationManager):
    def __init__(
        window_size: int = 40,
        should_truncate_results: bool = True,
        *,
        per_turn: bool | int = False,
        pin_first: int | None = None,
        proactive_compression: bool | ProactiveCompressionConfig | None = None,
    )
```

- `window_size`: max number of messages (not full turns — a turn is a user+assistant
  message pair, so `window_size=8` keeps roughly the last 4 ticks) kept in the agent's
  history before older messages are evicted. **This is the key lever.**
- For this game-tick use case, a **small window (e.g., `window_size=6`, i.e., ~3 prior
  ticks)** is sufficient — the point of memory here is opponent-tendency recall via
  AgentCore's long-term strategies (semantic/summarization), not literal short-term
  replay of raw coordinates from 30 seconds ago.
- `per_turn=True` is recommended so the window is enforced *before* every model call
  (not just at the end of the full agent loop), which matters here because Gateway tool
  calls can add extra turns within a single `agent()` invocation.

## Required code changes

### 1. `ai-team-strands-memory/memory_agent_base.py` (or the new combined variant's
   equivalent file)

Current code:

```python
from strands import Agent
from strands.models import BedrockModel
from bedrock_agentcore.memory.integrations.strands.session_manager import AgentCoreMemorySessionManager
from bedrock_agentcore.memory.integrations.strands.config import AgentCoreMemoryConfig


def create_memory_agent(
    system_prompt: str,
    player_id: int,
    position_label: str,
    model_id: str = "us.amazon.nova-micro-v1:0",
) -> Agent:
    memory_id = os.environ.get("MEMORY_ID")
    team_id = os.environ.get("TEAM_ID", "default-team")

    if not memory_id:
        raise RuntimeError("MEMORY_ID environment variable is required")

    session_manager = AgentCoreMemorySessionManager(
        agentcore_memory_config=AgentCoreMemoryConfig(
            memory_id=memory_id,
            session_id=f"match-{team_id}-{position_label}",
            actor_id=f"{team_id}-{position_label}",
        ),
        region_name=os.environ.get("AWS_DEFAULT_REGION"),
    )

    model = BedrockModel(model_id=model_id)
    return Agent(
        model=model,
        system_prompt=system_prompt,
        session_manager=session_manager,
    )
```

Required change — add a bounded `conversation_manager` alongside the existing
`session_manager`:

```python
from strands import Agent
from strands.models import BedrockModel
from strands.agent.conversation_manager import SlidingWindowConversationManager
from bedrock_agentcore.memory.integrations.strands.session_manager import AgentCoreMemorySessionManager
from bedrock_agentcore.memory.integrations.strands.config import AgentCoreMemoryConfig

# Keep only the last ~3 ticks (6 messages = 3 user/assistant pairs) in the prompt.
# Long-term recall (opponent tendencies, etc.) is handled by AgentCore Memory's
# long-term strategies (semantic/summarization), not by replaying raw history.
HISTORY_WINDOW_MESSAGES = 6


def create_memory_agent(
    system_prompt: str,
    player_id: int,
    position_label: str,
    model_id: str = "us.amazon.nova-micro-v1:0",
) -> Agent:
    memory_id = os.environ.get("MEMORY_ID")
    team_id = os.environ.get("TEAM_ID", "default-team")

    if not memory_id:
        raise RuntimeError("MEMORY_ID environment variable is required")

    session_manager = AgentCoreMemorySessionManager(
        agentcore_memory_config=AgentCoreMemoryConfig(
            memory_id=memory_id,
            session_id=f"match-{team_id}-{position_label}",
            actor_id=f"{team_id}-{position_label}",
        ),
        region_name=os.environ.get("AWS_DEFAULT_REGION"),
    )

    conversation_manager = SlidingWindowConversationManager(
        window_size=HISTORY_WINDOW_MESSAGES,
        per_turn=True,  # enforce the window before every model call, not just at the end
    )

    model = BedrockModel(model_id=model_id)
    return Agent(
        model=model,
        system_prompt=system_prompt,
        session_manager=session_manager,
        conversation_manager=conversation_manager,
    )
```

### 2. New combined gateway+memory agent factory

If/when the new combined variant is created (e.g.,
`ai-team-strands-gateway-memory/combined_agent_base.py` or similar — check with the team
for the actual planned file name/location), it must include **both** `tools=` (from the
gateway pattern) **and** `session_manager=` + `conversation_manager=` (from the corrected
memory pattern above) on the same `Agent` construction. Example shape:

```python
from strands import Agent
from strands.models import BedrockModel
from strands.agent.conversation_manager import SlidingWindowConversationManager
from strands.tools.mcp.mcp_client import MCPClient
from mcp.client.streamable_http import streamablehttp_client
from bedrock_agentcore.memory.integrations.strands.session_manager import AgentCoreMemorySessionManager
from bedrock_agentcore.memory.integrations.strands.config import AgentCoreMemoryConfig

HISTORY_WINDOW_MESSAGES = 6


def _create_gateway_transport():
    gateway_url = os.environ.get("GATEWAY_URL")
    if not gateway_url:
        raise RuntimeError("GATEWAY_URL environment variable is required")
    headers = {}
    access_token = os.environ.get("GATEWAY_ACCESS_TOKEN")
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"
    return streamablehttp_client(gateway_url, headers=headers)


def create_combined_agent(
    system_prompt: str,
    player_id: int,
    position_label: str,
    model_id: str = "us.amazon.nova-micro-v1:0",
) -> tuple[Agent, MCPClient]:
    memory_id = os.environ.get("MEMORY_ID")
    team_id = os.environ.get("TEAM_ID", "default-team")
    if not memory_id:
        raise RuntimeError("MEMORY_ID environment variable is required")

    session_manager = AgentCoreMemorySessionManager(
        agentcore_memory_config=AgentCoreMemoryConfig(
            memory_id=memory_id,
            session_id=f"match-{team_id}-{position_label}",
            actor_id=f"{team_id}-{position_label}",
        ),
        region_name=os.environ.get("AWS_DEFAULT_REGION"),
    )

    conversation_manager = SlidingWindowConversationManager(
        window_size=HISTORY_WINDOW_MESSAGES,
        per_turn=True,
    )

    mcp_client = MCPClient(_create_gateway_transport)
    model = BedrockModel(model_id=model_id)

    with mcp_client:
        tools = mcp_client.list_tools_sync()

    agent = Agent(
        model=model,
        system_prompt=system_prompt,
        tools=tools,
        session_manager=session_manager,
        conversation_manager=conversation_manager,
    )

    return agent, mcp_client
```

The invoke handler for this combined agent should follow the same pattern as
`gateway_invoke_handler.py` (wrap the `agent(state_summary)` call in
`with mcp_client:` so tools remain reachable during invocation).

## Validation after implementing

1. **Unit/local test**: run each position's `test_local.py` (pattern already exists in
   `ai-team-strands-memory/ai-gk/test_local.py` etc.) and confirm it still returns valid
   commands.
2. **Token check (primary success metric)**: after deploying, invoke the agent for a
   simulated multi-tick sequence (10+ ticks) and pull the `invoke_agent` span's
   `gen_ai.usage.prompt_tokens` from CloudWatch Logs (same query pattern used in the
   investigation — Logs Insights against the `/aws/bedrock-agentcore/runtimes/<agent>-*`
   log group, `spans` stream). Confirm `prompt_tokens` **stays roughly flat** across
   ticks (should hover in the low thousands, not grow into the tens/hundreds of
   thousands as ticks accumulate). This is the direct regression test for the root
   cause.
3. **Latency check**: confirm `strands.event_loop.latency` / `gen_ai.server.request.duration`
   metrics (same OTel metrics used in the investigation) do not regress compared to the
   `ai-team-strands-balanced` baseline, despite the added tool-schema overhead from
   gateway.
4. **Behavior check**: confirm the agent still plays sensibly — since `window_size=6`
   only removes *raw verbatim replay* of ticks older than ~3 turns, and does not touch
   `summarize_state()`'s per-tick content or AgentCore Memory's long-term
   strategies, decision quality for the *current* tick should be unaffected. If the team
   wants explicit validation that opponent-tendency recall (the actual reason memory was
   added) still works, test a scenario where the same simulated opponent behavior repeats
   across many ticks and confirm the agent's positioning adapts — this recall path goes
   through AgentCore Memory's long-term semantic/summarization strategies, not the raw
   sliding window, so it should be unaffected by this change.

## Tunable parameters (adjust based on validation results)

- `HISTORY_WINDOW_MESSAGES`: start at `6` (≈3 ticks). Increase only if validation shows
  the agent needs more short-term raw context than long-term memory strategies provide.
  Do not remove the cap entirely (i.e., do not fall back to unbounded) — that reintroduces
  the root cause this task fixes.
- `per_turn`: keep `True` for this workload (agents that may invoke gateway tools
  mid-turn, which adds extra messages within a single tick's `agent()` call).
