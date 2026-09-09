# Heavenly Emperrors Agents (Strands + Memory + Gateway) — Per-Position Soccer Agents

Five AI agents that each control a single player in a 5v5 soccer match, built with
[Strands Agents SDK](https://github.com/strands-agents/sdk-python) and deployed to
[Amazon Bedrock AgentCore](https://docs.aws.amazon.com/bedrock-agentcore/).

This team lines up in a defensive **1 forward, 1 midfielder, 2 defenders, 1 goalkeeper**
formation — a solid back two shielding the keeper, a single midfield link, and a lone
striker leading the line.

Every agent combines two capabilities on the same Strands `Agent`, wired up by
`create_combined_agent()`:

- **Memory** — backed by an [AgentCore Memory](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/memory.html)
  short-term memory (STM) session, so each agent recalls earlier ticks within a match and
  adapts to what it has seen (opponent tendencies, which passes/shots have worked, marking
  duties).
- **Gateway tactical tools** — MCP tools served via an
  [AgentCore Gateway](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway.html)
  (`find_open_space`, `evaluate_shot`, `calculate_pass_options`, `get_defensive_assignment`)
  that let the agent call out to Lambda-computed tactical analysis instead of reasoning
  about raw coordinates itself.

Both are independent `Agent` constructor arguments (`session_manager`/`conversation_manager`
vs. `tools`), so they compose without conflicting. Combining them naively would compound
the two together's worst case — unbounded conversation history *plus* tool-schema overhead
on every call — so `create_combined_agent()` also applies a bounded
`SlidingWindowConversationManager` (`window_size=6`, `per_turn=True`) to keep the per-tick
prompt flat across a match. See `combined_agent_base.py` for details.

## Architecture

```
agents/
├── lib/            # Shared library (single source of truth, used by all teams)
└── heavenly-emperrors-agents/
    ├── ai-gk/                    # Goalkeeper  (player 0) — Nova Pro
    ├── ai-def1/                  # Defender 1  (player 1) — Nova Lite
    ├── ai-def2/                  # Defender 2  (player 2) — Nova Lite
    ├── ai-mid/                   # Midfielder  (player 3) — Nova Pro
    ├── ai-fwd1/                  # Forward     (player 4) — Nova Lite
    ├── gateway_tools/            # Lambda implementations of the 4 tactical MCP tools
    ├── combined_agent_base.py     # create_combined_agent() — CDK deploy_all.py flow
    ├── combined_invoke_handler.py # create_combined_invoke_handler() — wraps calls in `with mcp_client:`
    ├── memory_agent_base.py       # create_memory_agent() — legacy deploy-all.sh flow (memory-only, MEMORY_ID)
    ├── memory_agent_base_cdk.py   # create_memory_agent() — CDK flow (memory-only, MEMORY_TEAM_MEMORY_ID)
    ├── create_memory.py           # One-time STM resource creator (legacy flow only)
    ├── deploy_all.py              # Build + deploy script (CDK flow, creates memory + gateway)
    ├── destroy_all.py             # Tear down agent runtimes
    └── README.md
```

Each agent has the same structure:

```
ai-<position>/
├── src/main.py          # Agent code
├── pyproject.toml       # Python dependencies
├── test_local.py        # Local tests (no AWS needed)
└── .gitignore
```

### How it works

Every agent's `main.py` follows the same pattern:

1. **System prompt** — tells the LLM what position it plays, what commands are available,
   how to use its memory of earlier ticks, and which tactical tools it has access to
2. **Fallback config** — rule-based behavior when the LLM fails to respond properly
3. **Wire it up** — `create_combined_agent()` (memory + gateway tools, team-level) +
   `create_combined_invoke_handler()` (wraps the call in `with mcp_client:` so Gateway
   tools stay reachable)

The shared `lib/` provides:
- `agent_base.py` — the non-memory, non-gateway `create_agent()` factory + `create_invoke_handler()`, used by other team variants
- `fallback.py` — configurable rule-based fallback per position
- `parsing.py` — extracts JSON commands from LLM responses
- `state.py` — summarizes game state into text for the LLM
- `_bootstrap.py` — resolves `lib/` path for both local dev and deployed environments
- `test_helpers.py` — mock AgentCore (`mock_agentcore_memory`, `mock_agentcore_gateway`) + sample game state for local tests

Memory+Gateway support lives at the team level (not in the shared lib), in
`combined_agent_base.py` — a single `create_combined_agent(system_prompt, player_id,
position_label, model_id)` factory used by the CDK flow (`deploy_all.py`):
- The `team_memory` resource is declared in `agentcore/agentcore.json`, created by the CDK
  stack, and its ID is injected into every runtime as `MEMORY_TEAM_MEMORY_ID`.
- The `tactical-tools` Gateway (and its 4 Lambda targets, built from `gateway_tools/`) is
  also declared in `agentcore/agentcore.json`, created by the CDK stack, and its endpoint
  is injected as `AGENTCORE_GATEWAY_TACTICAL_TOOLS_URL`.

`deploy_all.py` stages `combined_agent_base.py` and `combined_invoke_handler.py` into each
agent directory at deploy time, so `src/main.py` imports them unchanged.

The pre-existing `memory_agent_base.py` / `memory_agent_base_cdk.py` (memory-only, no
Gateway tools) remain in the repo for reference but are no longer wired into any agent's
`main.py`.


## Prerequisites

- Python 3.10+
- Node.js 20+ and npm
- AWS CLI configured with valid credentials
- AgentCore CLI and CDK: `npm install -g @aws/agentcore aws-cdk`
- [uv](https://docs.astral.sh/uv/getting-started/installation/) — used by the AgentCore CLI to package Python dependencies
- AWS account with Bedrock model access (Nova Micro, Lite, and/or Pro)
- CDK bootstrap (one-time per account/region): `cdk bootstrap aws://<account-id>/<region>`

Works on macOS, Linux, and Windows (PowerShell) — no WSL required.

## Quick Start

### 1. Run local tests (no AWS needed)

Local tests exercise state summary, parsing, and fallback with mocked AgentCore Memory
and Gateway clients (`mock_agentcore_memory`, `mock_agentcore_gateway`). No LLM, Memory,
or Gateway calls are made; deploy to AgentCore to test the real memory + tactical tools
integration.

```bash
# Test a single agent
python3 ai-gk/test_local.py

# Test all five agents
./test-all-players.sh
```

### 2. Deploy to AWS

```bash
# Deploy all 5 agents (and the team_memory resource + tactical-tools gateway) via CDK
AWS_DEFAULT_REGION=us-east-1 python deploy_all.py
```

On Windows (PowerShell):
```powershell
$env:AWS_DEFAULT_REGION="us-east-1"
python deploy_all.py
```

The deploy script:
1. Runs `cdk bootstrap` (idempotent — safe to run every time)
2. Temporarily copies the shared `lib/` directory and stages `combined_agent_base.py` +
   `combined_invoke_handler.py` into each agent's directory
3. Runs `agentcore deploy` once, which creates the `team_memory` STM resource and the
   `tactical-tools` Gateway (with its 4 Lambda targets built from `gateway_tools/`),
   grants each runtime access to both, and injects `MEMORY_TEAM_MEMORY_ID` and
   `AGENTCORE_GATEWAY_TACTICAL_TOOLS_URL`
4. Removes the injected files on success, failure, or Ctrl+C

The shared `lib/` and combined base module remain single sources of truth — the copies are
temporary and never committed.

To tear the agent runtimes back down, run `python destroy_all.py` (the memory resource and
gateway are left in place; remove their entries from `agentcore/agentcore.json` and
redeploy to delete them).


## Creating Your Own Agent

The easiest way is to copy an existing agent and modify it:

```bash
cp -r ai-gk ai-myagent
```

Then edit these files:

### `ai-myagent/src/main.py`

```python
# 1. Set which player this agent controls (0-4)
MY_PLAYER_ID = 0
POSITION_LABEL = "GK"

# 2. Write your system prompt — tell the LLM its role, available commands,
#    how to use its memory of earlier ticks, and which tactical tools it has
SYSTEM_PROMPT = f"""You are an AI soccer goalkeeper... You have MEMORY of previous ticks...
You have access to tactical analysis TOOLS via MCP..."""

# 3. Pick a fallback config (or create your own in lib/fallback.py)
fallback_commands = build_fallback(GK_CONFIG)

# 4. Choose your model — create_combined_agent backs the agent with AgentCore Memory
#    (STM) AND Gateway tactical tools (find_open_space, evaluate_shot,
#    calculate_pass_options, get_defensive_assignment)
agent, mcp_client = create_combined_agent(SYSTEM_PROMPT, MY_PLAYER_ID, POSITION_LABEL, model_id="us.amazon.nova-micro-v1:0")
create_combined_invoke_handler(
    app, agent, mcp_client, MY_PLAYER_ID, POSITION_LABEL, fallback_commands,
    fallback_cfg=GK_CONFIG,
)
```

### `agentcore/agentcore.json`

Add a new runtime entry for your agent (the `team_memory` resource and `tactical-tools`
gateway are already declared, so the CDK stack injects `MEMORY_TEAM_MEMORY_ID` and
`AGENTCORE_GATEWAY_TACTICAL_TOOLS_URL` into it automatically):

```json
{
  "name": "ai_myagent_memory_agent",
  "build": "CodeZip",
  "entrypoint": "src/main.py",
  "codeLocation": "ai-myagent/",
  "runtimeVersion": "PYTHON_3_12",
  "networkMode": "PUBLIC",
  "protocol": "HTTP"
}
```

### `deploy_all.py`

Add your agent to the `ALL_AGENTS` list:

```python
ALL_AGENTS = ["ai-gk", "ai-def1", "ai-mid", "ai-def2", "ai-fwd1", "ai-myagent"]
```


## Player IDs and Positions

| Player ID | Position    | Directory | Default Model |
|-----------|-------------|-----------|---------------|
| 0         | Goalkeeper  | ai-gk     | Nova Pro      |
| 1         | Defender 1  | ai-def1   | Nova Lite     |
| 2         | Defender 2  | ai-def2   | Nova Lite     |
| 3         | Midfielder  | ai-mid    | Nova Pro      |
| 4         | Forward     | ai-fwd1   | Nova Lite     |

## Tactical Tools (AgentCore Gateway MCP)

Every agent has access to the same four MCP tools, served via the `tactical-tools`
AgentCore Gateway and implemented as Lambda functions in `gateway_tools/`. Each system
prompt tells its agent which of these are most relevant to its position, but any agent
may call any tool:

| Tool | What it does |
|---|---|
| `calculate_pass_options` | Calculates pass success probability for each teammate based on interception risk |
| `evaluate_shot` | Evaluates shot success probability and recommends an aim point |
| `find_open_space` | Grid-based open space finder by zone (`attack`, `midfield`, `defense`) |
| `get_defensive_assignment` | Ranks opponent threats for defensive marking priority |

Tool calls happen inside the model's own reasoning loop — the agent decides when to call
a tool based on its system prompt, then incorporates the tool's result into its next
response. `combined_invoke_handler.py` wraps each tick's `agent()` call in
`with mcp_client:` so the Gateway connection is live for the whole invocation, including
any mid-turn tool calls.

## Available Commands

Commands are what the LLM returns to control the player each tick.

**One-shot** (execute once):
- `MOVE_TO` — target_x, target_y, sprint
- `PASS` — target_player_id, type (GROUND/AERIAL/THROUGH)
- `SHOOT` — aim_location (TL/TR/BL/BR/CENTER), power (0.0-1.0)
- `GK_DISTRIBUTE` — target_player_id, method (THROW/KICK)

**Maintained** (persist across ticks):
- `PRESS_BALL` — intensity (0.0-1.0)
- `MARK` — target_player_id, tightness (LOOSE/TIGHT)
- `INTERCEPT` — aggressive (bool)
- `FOLLOW_PLAYER` — target_player_id, target_team, distance

**Tactical**:
- `SET_STANCE` — stance (0=Balanced, 1=Attack, 2=Defend)
- `CLEAR_OVERRIDE` — return to default AI

## Error Handling

Each agent has three layers of fallback:

1. **LLM response** — parsed into commands via `lib/parsing.py`
2. **Rule-based fallback** — position-specific logic from `lib/fallback.py`
3. **Last-resort command** — a single safe command (e.g., SET_STANCE) when everything else fails

### Models that write Python instead of JSON

Models are trained on a lot of Python, so they will occasionally give you Python's spelling
of a value rather than JSON's:

```json
[{"commandType": "MOVE_TO", "parameters": {"target_x": 2.0, "sprint": True}}]
```

`True` is valid Python and invalid JSON. Strictly parsed, that whole command is discarded
and your agent drops to layer 2 — which looks like nothing is wrong: no crash, no error,
just an agent that has quietly stopped using its model. The only clue is a
`LLM parse failed` line in its log.

`lib/json_tolerant.py` handles this. **Only after a strict parse has already failed**, it
retries on a normalised copy, recovering:

- bare `True` / `False` / `None` → `true` / `false` / `null`
- a trailing comma before `}` or `]`
- a markdown code fence wrapping the payload

Valid JSON never reaches that code, so well-formed output behaves exactly as before. Text
inside strings is never rewritten — `{"note": "True story"}` comes back untouched.

You don't need to do anything to get this; it is already wired into `lib/parsing.py`. If
you want to see how often your model needs it, pass a callback:

```python
parse_commands(response_text, team_id, my_player_id, lambda raw: log.warn(f"recovered: {raw[:200]}"))
```

Run `python3 lib/test_parsing.py` to see the cases it covers.

If your model does this a lot, it is worth tightening your system prompt — recovery is a
safety net, not a substitute for asking clearly for JSON.

## Field Coordinates

- x: roughly -55 to +55
- y: roughly -35 to +35
- Team 0 (HOME) defends -x, attacks toward +x
- Team 1 (AWAY) defends +x, attacks toward -x
