# Heavenly Emperrors Agents (Strands + Memory) — Per-Position Soccer Agents

Five AI agents that each control a single player in a 5v5 soccer match, built with
[Strands Agents SDK](https://github.com/strands-agents/sdk-python) and deployed to
[Amazon Bedrock AgentCore](https://docs.aws.amazon.com/bedrock-agentcore/).

This team lines up in a defensive **1 forward, 1 midfielder, 2 defenders, 1 goalkeeper**
formation — a solid back two shielding the keeper, a single midfield link, and a lone
striker leading the line.

Every agent is **memory-aware**: each is created with `create_memory_agent()`, which
backs the Strands `Agent` with an [AgentCore Memory](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/memory.html)
short-term memory (STM) session so it recalls earlier ticks within a match and adapts to
what it has seen (opponent tendencies, which passes/shots have worked, marking duties).

## Architecture

```
agents/
├── lib/            # Shared library (single source of truth, used by all teams)
└── heavenly-emperrors-agents/
    ├── ai-gk/                  # Goalkeeper  (player 0) — Nova Micro
    ├── ai-def1/                # Defender 1  (player 1) — Nova Lite
    ├── ai-def2/                # Defender 2  (player 2) — Nova Lite
    ├── ai-mid/                 # Midfielder  (player 3) — Nova Pro
    ├── ai-fwd1/                # Forward     (player 4) — Nova Lite
    ├── memory_agent_base.py     # create_memory_agent() — legacy deploy-all.sh flow (MEMORY_ID)
    ├── memory_agent_base_cdk.py # create_memory_agent() — CDK deploy_all.py flow (MEMORY_TEAM_MEMORY_ID)
    ├── create_memory.py         # One-time STM resource creator (legacy flow only)
    ├── deploy_all.py            # Build + deploy script (CDK flow, creates memory)
    ├── destroy_all.py           # Tear down agent runtimes
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

1. **System prompt** — tells the LLM what position it plays, what commands are available, and how to use its memory of earlier ticks
2. **Fallback config** — rule-based behavior when the LLM fails to respond properly
3. **Wire it up** — `create_memory_agent()` (from the team-level `memory_agent_base`) + `create_invoke_handler()` (from the shared lib)

The shared `lib/` provides:
- `agent_base.py` — invoke handler with 3-layer error handling (LLM → fallback → last-resort), plus the non-memory `create_agent()` factory
- `fallback.py` — configurable rule-based fallback per position
- `parsing.py` — extracts JSON commands from LLM responses
- `state.py` — summarizes game state into text for the LLM
- `_bootstrap.py` — resolves `lib/` path for both local dev and deployed environments
- `test_helpers.py` — mock AgentCore (`mock_agentcore_memory`) + sample game state for local tests

Memory support lives at the team level (not in the shared lib), in two sibling modules
with the same `create_memory_agent(system_prompt, player_id, position_label, model_id)` API:
- `memory_agent_base_cdk.py` — used by the CDK flow (`deploy_all.py`). The `team_memory`
  resource is declared in `agentcore/agentcore.json`, created by the CDK stack, and its ID
  is injected into every runtime as `MEMORY_TEAM_MEMORY_ID`.
- `memory_agent_base.py` — used by the legacy flow (`deploy-all.sh`), which reads a
  `MEMORY_ID` env var created via `create_memory.py`.

`deploy_all.py` stages `memory_agent_base_cdk.py` into each agent directory as
`memory_agent_base.py` at deploy time, so `src/main.py` imports it unchanged.


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

Local tests exercise state summary, parsing, and fallback with a mocked AgentCore Memory
(`mock_agentcore_memory`). No LLM or Memory calls are made; deploy to AgentCore to test
the real memory integration.

```bash
# Test a single agent
python3 ai-gk/test_local.py

# Test all five agents
./test-all-players.sh
```

### 2. Deploy to AWS

```bash
# Deploy all 5 agents (and the team_memory resource) via CDK
AWS_DEFAULT_REGION=us-east-1 python deploy_all.py
```

On Windows (PowerShell):
```powershell
$env:AWS_DEFAULT_REGION="us-east-1"
python deploy_all.py
```

The deploy script:
1. Runs `cdk bootstrap` (idempotent — safe to run every time)
2. Temporarily copies the shared `lib/` directory and stages `memory_agent_base_cdk.py`
   as `memory_agent_base.py` into each agent's directory
3. Runs `agentcore deploy` once, which creates the `team_memory` STM resource, grants each
   runtime access to it, and injects its ID as `MEMORY_TEAM_MEMORY_ID`
4. Removes the injected files on success, failure, or Ctrl+C

The shared `lib/` and memory base module remain single sources of truth — the copies are
temporary and never committed.

To tear the agent runtimes back down, run `python destroy_all.py` (the memory resource is
left in place; remove its entry from `agentcore/agentcore.json` and redeploy to delete it).


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
#    and how to use its memory of earlier ticks
SYSTEM_PROMPT = f"""You are an AI soccer goalkeeper... You have MEMORY of previous ticks..."""

# 3. Pick a fallback config (or create your own in lib/fallback.py)
fallback_commands = build_fallback(GK_CONFIG)

# 4. Choose your model — create_memory_agent backs the agent with AgentCore Memory (STM)
agent = create_memory_agent(SYSTEM_PROMPT, MY_PLAYER_ID, POSITION_LABEL, model_id="us.amazon.nova-micro-v1:0")
```

### `agentcore/agentcore.json`

Add a new runtime entry for your agent (the `team_memory` resource is already declared
under `memories`, so the CDK stack injects `MEMORY_TEAM_MEMORY_ID` into it automatically):

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
| 0         | Goalkeeper  | ai-gk     | Nova Micro    |
| 1         | Defender 1  | ai-def1   | Nova Lite     |
| 2         | Defender 2  | ai-def2   | Nova Lite     |
| 3         | Midfielder  | ai-mid    | Nova Pro      |
| 4         | Forward     | ai-fwd1   | Nova Lite     |

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
