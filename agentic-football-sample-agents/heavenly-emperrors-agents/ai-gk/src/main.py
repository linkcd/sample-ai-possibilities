"""
AI Soccer Goalkeeper Agent (Memory + Gateway, Solid Defensive) — Controls ONLY player 0 (Goalkeeper).
Stays deep, holds the line between ball and goal, and distributes to whichever of the
midfielder (3) or forward (4) is furthest up the pitch.
Uses Strands SDK + Amazon Nova Pro + AgentCore Memory for cross-tick recall,
plus AgentCore Gateway MCP tactical tools.
"""

import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib")); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lib"))
from _bootstrap import setup_lib_path; setup_lib_path(__file__)

# combined_agent_base lives one level above src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from combined_agent_base import create_combined_agent
from combined_invoke_handler import create_combined_invoke_handler
from fallback import build_fallback, GK_CONFIG

app = BedrockAgentCoreApp()

# --- Position Config ---
MY_PLAYER_ID = 0
POSITION_LABEL = "GK"

# --- System Prompt ---

SYSTEM_PROMPT = f"""You are the GOALKEEPER on a 5v5 football team, Player {MY_PLAYER_ID}. You may be on the HOME or AWAY team — read "Team" and the goal positions from the game state each tick.

Field sides (from the state): "Your goal" is at one end (my_goal_x) and the "Opponent goal" at the other (opp_goal_x). If HOME your goal is at x=-55 and you attack toward +x; if AWAY your goal is at x=+55 and you attack toward -x. "Up the pitch" always means toward the opponent's goal.

You have access to tactical analysis TOOLS via MCP. Use them to make better decisions:
- Use `get_defensive_assignment` to identify the most dangerous opponent to stay goal-side of
- Use `calculate_pass_options` when you have the ball to find the best distribution target

Your role: defend your goal — stay deep near your own goal line and keep yourself on the line between the ball and the centre of your goal.

You receive the game state each tick and return exactly ONE command for Player {MY_PLAYER_ID} only.

Decision priorities (in order):
1. If you have the ball → GK_DISTRIBUTE to whichever of the midfielder (3) or forward (4) is furthest up the pitch (closest to the opponent goal). Use KICK if that teammate is past the halfway line, otherwise THROW.
2. If the ball is loose within ~5 units of you → INTERCEPT (aggressive: false).
3. Otherwise → MOVE_TO onto the line between the ball and the centre of your own goal (x very close to your own goal line, y tracking the ball), sprint: false.

Memory: use recalled ticks to remember the opponent's most dangerous shooters and which outlet (3 or 4) has been open, and adjust your side-to-side positioning.

You must NEVER:
- Advance far from your own goal or push up the pitch toward the opponent's goal.
- SHOOT, PASS as an outfield player, or join the attack.
- Sprint except for a direct, close-range save or interception.

## Available Commands (commandType → parameters)

ONE-SHOT:
- MOVE_TO: target_x (float), target_y (float), sprint (bool) — stay on/near your goal line only
- GK_DISTRIBUTE: target_player_id (int), method ("THROW"|"KICK") — your primary tool when you have the ball
- SLIDE_TACKLE: target_player_id (int), sprint (bool), distance (float) — last resort only

MAINTAINED:
- INTERCEPT: aggressive (bool) — set false; only intercept balls very close to you
- FOLLOW_PLAYER: target_player_id (int), target_team ("HOME"|"AWAY"), distance (float)

TACTICAL:
- SET_STANCE: stance (0=Balanced, 1=Attack, 2=Defend) — you play stance 2 (Defend)
- CLEAR_OVERRIDE: {{}} — return to default AI
- RESET: {{}} — clear all overrides for team

## Field
- Coordinates: x roughly -55 to +55, y roughly -35 to +35
- Team 0 (HOME) defends -x, attacks toward +x
- Team 1 (AWAY) defends +x, attacks toward -x

## Response
Return ONLY a JSON array with exactly ONE command for player {MY_PLAYER_ID}.
Example (distribute to the forward when they are furthest up): [{{"commandType":"GK_DISTRIBUTE","playerId":{MY_PLAYER_ID},"parameters":{{"target_player_id":4,"method":"KICK"}},"duration":0}}]
Example (hold the line between ball and goal — use a target_x very close to YOUR OWN goal line, i.e. near my_goal_x): [{{"commandType":"MOVE_TO","playerId":{MY_PLAYER_ID},"parameters":{{"target_x":-50.0,"target_y":3.0,"sprint":false}},"duration":0}}]
Return ONLY the JSON array, no text before or after."""


# --- Fallback ---
#
# GK_CONFIG already gives the solid-defensive positioning we want on the rule-based
# fallback path: it keeps the keeper deep (default_x_factor 0.9 toward our own goal),
# tracks the ball laterally, and holds stance 2 (Defend). We only override the on-ball
# behaviour so distribution goes to whichever of the midfielder (3) or forward (4) is
# FURTHEST up the pitch — matching the system prompt — instead of the shared lib's
# "nearest teammate" default.

from state import get_goal_positions, _is_my_team, _player_idx

_base_fallback = build_fallback(GK_CONFIG)

# Distribution outlets, most preferred first: midfielder (3), then forward (4).
_DISTRIBUTION_OUTLETS = (3, 4)


def fallback_commands(game_state: dict, team_id: int, my_player_id: int) -> list[dict]:
    ball = game_state.get("ball", {})
    players = game_state.get("players", [])

    # Determine whether THIS keeper is the ball holder.
    possession_agent = ball.get("possessionAgentId")
    keeper_has_ball = possession_agent == f"agentId_{my_player_id}"

    if keeper_has_ball:
        my_goal_x, opp_goal_x = get_goal_positions(team_id)
        # "Furthest up the pitch" = closest to the opponent goal x.
        candidates = [
            p for p in players
            if _is_my_team(p, team_id) and _player_idx(p) in _DISTRIBUTION_OUTLETS
        ]
        if candidates:
            target = min(
                candidates,
                key=lambda p: abs(p.get("position", {}).get("x", 0) - opp_goal_x),
            )
            target_id = _player_idx(target)
            target_x = target.get("position", {}).get("x", 0)
            # KICK long if the outlet is past the halfway line, else a safer THROW.
            past_halfway = abs(target_x) < abs(opp_goal_x) and (
                (opp_goal_x > 0 and target_x > 0) or (opp_goal_x < 0 and target_x < 0)
            )
            method = "KICK" if past_halfway else "THROW"
            return [{
                "commandType": "GK_DISTRIBUTE",
                "playerId": my_player_id,
                "teamId": team_id,
                "parameters": {"target_player_id": target_id, "method": method},
                "duration": 0,
            }]

    # Not on the ball (or no outlet found): use the deep-defensive base behaviour.
    return _base_fallback(game_state, team_id, my_player_id)


# --- Wire it up ---

agent, mcp_client = create_combined_agent(SYSTEM_PROMPT, MY_PLAYER_ID, POSITION_LABEL, model_id="us.amazon.nova-pro-v1:0")
create_combined_invoke_handler(
    app, agent, mcp_client, MY_PLAYER_ID, POSITION_LABEL, fallback_commands,
    fallback_cfg=GK_CONFIG,
)

if __name__ == "__main__":
    app.run()
