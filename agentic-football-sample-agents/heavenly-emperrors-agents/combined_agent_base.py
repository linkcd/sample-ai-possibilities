"""Combined memory+gateway agent factory for AI soccer position agents (CDK/npm-CLI flow).

Merges the two team-level agent factories that used to be deployed as separate
teams:
  - memory_agent_base_cdk.py   — AgentCore Memory (STM) for cross-tick recall
  - ai-team-strands-gateway/gateway_agent_base_cdk.py — AgentCore Gateway MCP
    tactical tools (calculate_pass_options, evaluate_shot, find_open_space,
    get_defensive_assignment)

Both additions are independent Strands `Agent` constructor arguments
(`session_manager`/`conversation_manager` vs. `tools`), so they compose on
the same Agent without conflicting.

IMPORTANT — bounded conversation window:
AgentCoreMemorySessionManager persists every tick's user/assistant turn to
AgentCore Memory and reloads that history on each invocation. Without a cap,
the full accumulated history is resent to the model on every tick (see
LATENCY_FIX_INSTRUCTIONS.md for the root-cause investigation). Combining
memory with Gateway tools makes this worse if left unbounded: every call
would carry both the growing history *and* the tool-schema token overhead
from the tools list. SlidingWindowConversationManager caps how many messages
are actually replayed into the model call, independent of how many are
stored in AgentCore Memory and independent of how many extra turns a Gateway
tool call adds within one invocation (`per_turn=True` re-applies the cap
before every model call, including mid-turn tool-result round trips).

Long-term recall (opponent tendencies, etc.) is provided by AgentCore
Memory's own long-term strategies, not by raw replay, so this cap does not
reduce recall quality — it only stops the prompt from growing unbounded.

Env vars (CDK flow, injected by agentcore/agentcore.json resources):
  MEMORY_TEAM_MEMORY_ID                 — team_memory resource (memories[])
  AGENTCORE_GATEWAY_TACTICAL_TOOLS_URL  — tactical-tools gateway (agentCoreGateways[])
  GATEWAY_ACCESS_TOKEN                  — optional bearer token (NONE auth by default)
  TEAM_ID                               — optional; used as actor_id/session_id prefix
"""

import os
from strands import Agent
from strands.models import BedrockModel
from strands.agent.conversation_manager import SlidingWindowConversationManager
from strands.tools.mcp.mcp_client import MCPClient
# The Streamable HTTP transport factory was renamed across mcp versions:
# older releases expose `streamablehttp_client`, newer ones `streamable_http_client`.
# Import whichever exists so this works regardless of the installed mcp version.
try:
    from mcp.client.streamable_http import streamablehttp_client
except ImportError:  # pragma: no cover - depends on installed mcp version
    from mcp.client.streamable_http import streamable_http_client as streamablehttp_client
from bedrock_agentcore.memory.integrations.strands.session_manager import AgentCoreMemorySessionManager
from bedrock_agentcore.memory.integrations.strands.config import AgentCoreMemoryConfig

# Maximum number of messages (user+assistant) replayed into the model prompt
# per call. 6 messages ≈ the last 3 ticks. Long-term recall of opponent
# tendencies, etc. is provided by AgentCore Memory's long-term strategies,
# not by raw replay, so this does not reduce agent intelligence — it only
# stops the prompt from growing unbounded across a match.
HISTORY_WINDOW_MESSAGES = 6


def _create_gateway_transport():
    """Build a Streamable HTTP transport pointing at the AgentCore Gateway.

    Supports both NONE auth (no token) and token-based auth.
    """
    gateway_url = os.environ.get("AGENTCORE_GATEWAY_TACTICAL_TOOLS_URL")
    if not gateway_url:
        raise RuntimeError(
            "AGENTCORE_GATEWAY_TACTICAL_TOOLS_URL environment variable is "
            "required. It is injected automatically when the agent is deployed "
            "via `python deploy_all.py` with the tactical-tools gateway "
            "declared in agentcore/agentcore.json."
        )

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
) -> "tuple[Agent, MCPClient | None]":
    """Create a Strands Agent backed by AgentCore Memory (STM) *and* Gateway
    MCP tactical tools, with a bounded conversation window.

    Required env vars:
      MEMORY_TEAM_MEMORY_ID                 — injected by the CDK stack for
                                               the "team_memory" resource in
                                               agentcore/agentcore.json
      AGENTCORE_GATEWAY_TACTICAL_TOOLS_URL  — injected by the CDK stack for
                                               the "tactical-tools" gateway in
                                               agentcore/agentcore.json
      TEAM_ID                               — used as actor_id and
                                               session_id prefix (optional;
                                               each team deploys into its own
                                               account, so the default cannot
                                               collide)

    AGENTCORE_GATEWAY_TACTICAL_TOOLS_URL is OPTIONAL: when unset, the agent is
    built memory-only (no MCP tools) instead of raising.

    Returns:
      (agent, mcp_client) — mcp_client is an MCPClient when Gateway tools are
      attached, or None when running memory-only. When it is not None the caller
      must use `with mcp_client:` while invoking the agent so tools stay available.
    """
    # Accept either memory-flow env var:
    #   MEMORY_TEAM_MEMORY_ID — injected by the CDK flow (python deploy_all.py)
    #   MEMORY_ID             — injected by the legacy flow (./deploy-all.sh)
    memory_id = os.environ.get("MEMORY_TEAM_MEMORY_ID") or os.environ.get("MEMORY_ID")
    team_id = os.environ.get("TEAM_ID", "default-team")

    if not memory_id:
        raise RuntimeError(
            "A memory resource ID is required. Set MEMORY_TEAM_MEMORY_ID "
            "(injected by `python deploy_all.py` via the team_memory resource in "
            "agentcore/agentcore.json) or MEMORY_ID (injected by `./deploy-all.sh`, "
            "created via create_memory.py)."
        )

    session_manager = AgentCoreMemorySessionManager(
        agentcore_memory_config=AgentCoreMemoryConfig(
            memory_id=memory_id,
            session_id=f"match-{team_id}-{position_label}",
            actor_id=f"{team_id}-{position_label}",
        ),
        # The CDK flow injects no region env var of its own; the AgentCore
        # runtime provides AWS_REGION. None lets boto3 resolve as a last resort.
        region_name=os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION"),
    )

    # Bound the per-call prompt to a small sliding window of recent ticks so
    # input tokens stay flat across a match instead of growing unbounded.
    # per_turn=True enforces the cap before every model call, including any
    # extra turns a Gateway tool call adds within a single invocation.
    conversation_manager = SlidingWindowConversationManager(
        window_size=HISTORY_WINDOW_MESSAGES,
        per_turn=True,
    )

    model = BedrockModel(model_id=model_id)

    # Gateway tactical tools are OPTIONAL. If AGENTCORE_GATEWAY_TACTICAL_TOOLS_URL
    # is set (CDK flow with the tactical-tools gateway deployed), attach the MCP
    # tools. If it is NOT set (e.g. the legacy ./deploy-all.sh flow, which does not
    # create a gateway), run memory-only: the agent still works via the LLM,
    # AgentCore Memory, and the rule-based fallback — it just has no MCP tools.
    gateway_url = os.environ.get("AGENTCORE_GATEWAY_TACTICAL_TOOLS_URL")
    if gateway_url:
        mcp_client = MCPClient(_create_gateway_transport)
        # Fetch tool definitions inside the context so the connection is active.
        with mcp_client:
            tools = mcp_client.list_tools_sync()
    else:
        mcp_client = None
        tools = []

    agent = Agent(
        model=model,
        system_prompt=system_prompt,
        tools=tools,
        session_manager=session_manager,
        conversation_manager=conversation_manager,
    )

    return agent, mcp_client
