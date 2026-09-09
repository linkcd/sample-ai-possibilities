"""Invoke handler for combined memory+gateway agents.

Same 3-layer error handling as the shared lib's agent_base.create_invoke_handler
(LLM -> fallback -> last-resort), but wraps the agent call inside the
MCPClient context manager so Gateway tools are available during invocation
(same requirement as gateway_invoke_handler.py in ai-team-strands-gateway).
"""

import json
from typing import Callable
from strands import Agent
from strands.tools.mcp.mcp_client import MCPClient

from parsing import parse_commands
from state import summarize_state
from fallback import FallbackConfig, build_last_resort


def create_combined_invoke_handler(
    app,
    agent: Agent,
    mcp_client: MCPClient,
    my_player_id: int,
    position_label: str,
    fallback_fn: Callable[[dict, int, int], list[dict]],
    fallback_cfg: FallbackConfig,
):
    """Create and register the @app.entrypoint invoke handler.

    Three layers of error handling, from best to worst:
      1. LLM response (with Gateway tool calls + Memory recall) -> parse into commands
      2. fallback_fn(game_state, team_id, my_player_id) -> rule-based commands
      3. last-resort command from fallback_cfg -> single safe command
    """
    log = app.logger
    last_resort = build_last_resort(fallback_cfg, my_player_id)

    @app.entrypoint
    async def invoke(payload, context):
        try:
            prompt = payload.get("prompt", "{}")
            prompt_data = json.loads(prompt) if isinstance(prompt, str) else prompt

            game_state = prompt_data.get("gameState", {})
            team_id = prompt_data.get("teamId", 0)

            # Honor myPlayers from payload if present, otherwise use configured player ID
            my_players = prompt_data.get("myPlayers", [my_player_id])
            effective_pid = my_players[0] if my_players else my_player_id

            state_summary = summarize_state(
                game_state, team_id, effective_pid, position_label
            )
            log.info(f"{position_label} combined agent invoked for team {team_id}, controlling player {effective_pid}")

            # Use MCP client context so Gateway tools are available while the
            # agent (and its Memory-backed session manager) runs.
            with mcp_client:
                response = agent(state_summary)
            response_text = str(response)

            def on_recovered(raw: str) -> None:
                # The model wrote Python-flavoured JSON (usually `True`/`False`/`None`).
                # We recovered it rather than dropping the command and falling back —
                # logged so you can see how often your model does this.
                log.warn(f"{position_label} recovered malformed JSON from the model: {raw[:200]}")

            commands = parse_commands(response_text, team_id, effective_pid, on_recovered)

            if commands:
                log.info(f"LLM+tools returned {len(commands)} commands: "
                         f"{[c.get('commandType') for c in commands]}")
                yield json.dumps(commands)
            else:
                log.warn(f"LLM parse failed, using fallback. Response: {response_text[:200]}")
                commands = fallback_fn(game_state, team_id, effective_pid)
                log.info(f"Fallback returned {len(commands)} commands")
                yield json.dumps(commands)

        except Exception as e:
            log.error(f"{position_label} combined agent error: {e}")
            try:
                prompt_data = json.loads(payload.get("prompt", "{}"))
                team_id = prompt_data.get("teamId", 0)
                my_players = prompt_data.get("myPlayers", [my_player_id])
                effective_pid = my_players[0] if my_players else my_player_id
                commands = fallback_fn(
                    prompt_data.get("gameState", {}),
                    team_id,
                    effective_pid,
                )
                yield json.dumps(commands)
            except Exception:
                cmd = dict(last_resort)
                cmd["teamId"] = 0  # best guess when payload parsing also failed
                yield json.dumps([cmd])

    return invoke
