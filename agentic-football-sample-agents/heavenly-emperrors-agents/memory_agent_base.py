"""Memory-aware agent factory for AI soccer position agents.

Extends the shared lib's agent_base with AgentCoreMemorySessionManager
so each agent persists conversation history across game ticks.

IMPORTANT — bounded history window:
AgentCoreMemorySessionManager persists every tick's user/assistant turn to
AgentCore Memory and reloads that history on each invocation. Without a cap,
the full accumulated history is resent to the model on every tick, which has
been observed to grow prompt size into the hundreds of thousands of tokens
over the course of a match (see LATENCY_FIX_INSTRUCTIONS.md for the root-cause
investigation). SlidingWindowConversationManager caps how many of those
messages are actually replayed into the model call, independent of how many
are stored in AgentCore Memory — long-term recall (e.g., opponent tendencies)
is handled by AgentCore Memory's own long-term strategies, not by verbatim
replay of raw history, so this cap does not reduce recall quality.
"""

import os
from strands import Agent
from strands.models import BedrockModel
from strands.agent.conversation_manager import SlidingWindowConversationManager
from bedrock_agentcore.memory.integrations.strands.session_manager import AgentCoreMemorySessionManager
from bedrock_agentcore.memory.integrations.strands.config import AgentCoreMemoryConfig

# Maximum number of messages (user+assistant) replayed into the model prompt
# per call. 6 messages ≈ the last 3 ticks. Long-term recall of opponent
# tendencies, etc. is provided by AgentCore Memory's long-term strategies,
# not by raw replay, so this does not reduce agent intelligence — it only
# stops the prompt from growing unbounded across a match.
HISTORY_WINDOW_MESSAGES = 6


def create_memory_agent(
    system_prompt: str,
    player_id: int,
    position_label: str,
    model_id: str = "us.amazon.nova-micro-v1:0",
) -> Agent:
    """Create a Strands Agent backed by AgentCore Memory (STM).

    Required env vars:
      MEMORY_ID  — AgentCore Memory resource ID
      TEAM_ID    — used as actor_id and session_id prefix
    """
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

    # Bound the per-call prompt to a small sliding window of recent ticks so
    # input tokens stay flat across a match instead of growing unbounded.
    # per_turn=True enforces the cap before every model call.
    conversation_manager = SlidingWindowConversationManager(
        window_size=HISTORY_WINDOW_MESSAGES,
        per_turn=True,
    )

    model = BedrockModel(model_id=model_id)
    return Agent(
        model=model,
        system_prompt=system_prompt,
        session_manager=session_manager,
        conversation_manager=conversation_manager,
    )
