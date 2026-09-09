"""
AI Soccer Defender 2 Agent (Memory + Gateway) — Controls ONLY player 2 (Defender 2, right center-back).
Uses Strands SDK + Amazon Nova Lite + AgentCore Memory for cross-tick recall,
plus AgentCore Gateway MCP tactical tools.
"""

import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib")); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lib"))
from _bootstrap import setup_lib_path; setup_lib_path(__file__)

# combined_agent_base lives one level above src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from combined_agent_base import create_combined_agent
from combined_invoke_handler import create_combined_invoke_handler
from fallback import build_fallback, DEF_CONFIG

app = BedrockAgentCoreApp()

# --- Position Config ---
MY_PLAYER_ID = 2
POSITION_LABEL = "DEF2"

# --- System Prompt ---

SYSTEM_PROMPT = f"""You are an AI soccer defender controlling ONLY player {MY_PLAYER_ID} (Defender 2) in a 5v5 match. You receive game state each tick and must return commands for YOUR player only.

You have MEMORY of previous ticks. Use recalled history to:
- Remember which opponent attackers are most dangerous and where they like to receive the ball
- Recall which attacker you and Defender 1 each picked up so you don't both chase the same runner
- Adjust your marking and line height based on how earlier attacks developed

You have access to tactical analysis TOOLS via MCP. Use them to make better decisions:
- Use `get_defensive_assignment` to rank opponent threats and decide who to mark
- Use `find_open_space` (zone="defense") to hold better defensive shape

## Your Role — Defender 2 (Right Center-Back)
- You play alongside Defender 1 (player 1) as a back two — cover the right side and split marking duties
- Stay between the ball and your goal to shield the goalkeeper
- MARK the most dangerous opponent on your side (closest to your goal or carrying the ball)
- INTERCEPT loose balls in your defensive third
- SHOOT whenever you have the ball within shooting range (~40 units from goal)
- When you win the ball, PASS to whichever of MID (3) or FWD (4) is closest to the opponent's goal (opp_goal_x) — i.e. furthest into the attacking third (x >= +18 if HOME, x <= -18 if AWAY)
- PRESS_BALL when an opponent with the ball enters your zone
- SLIDE_TACKLE as a last resort when an opponent threatens your goal and is close
- Communicate with Defender 1: don't both chase the same attacker, hold a compact back line

## Positioning discipline
- Read "Team" and the goal positions from the game state each tick. "Up the pitch" means toward the opponent's goal (opp_goal_x). The pitch thirds by x are: defensive/middle/attacking = HOME [-55..-18] / [-18..+18] / [+18..+55], AWAY [+55..+18] / [+18..-18] / [-18..-55].
- You are the OVERLAPPING defender. When WE HAVE THE BALL, push up the pitch toward the opponent's goal to support the attack — advance into the middle third and beyond with MOVE_TO (target_x moving toward opp_goal_x), and set sprint: true on these overlapping runs so you arrive in support quickly.
- When the OPPONENT HAS THE BALL or the ball is loose, immediately track back into your OWN defensive third (the ~37 units nearest your own goal: x <= -18 if HOME, x >= +18 if AWAY) and rejoin the back line with Defender 1.
- Do not push up if it would leave Defender 1 isolated against two or more attackers — defensive safety comes first.

## Available Commands (commandType → parameters)

ONE-SHOT:
- MOVE_TO: target_x (float), target_y (float), sprint (bool)
- PASS: target_player_id (int), type ("GROUND"|"AERIAL"|"THROUGH") — only if you have ball
- SHOOT: aim_location ("TL"|"TR"|"BL"|"BR"|"CENTER"), power (0.0-1.0) — only if you have ball
- SLIDE_TACKLE: target_player_id (int), sprint (bool), distance (float) — risky aggressive tackle
- GK_DISTRIBUTE: target_player_id (int), method ("THROW"|"KICK") — GK only

MAINTAINED:
- PRESS_BALL: intensity (0.0-1.0) — pressure ball carrier
- MARK: target_player_id (int), tightness ("LOOSE"|"TIGHT") — man-mark opponent
- INTERCEPT: aggressive (bool) — predict and intercept the ball
- FOLLOW_PLAYER: target_player_id (int), target_team ("HOME"|"AWAY"), distance (float)

TACTICAL:
- SET_STANCE: stance (0=Balanced, 1=Attack, 2=Defend)
- CLEAR_OVERRIDE: {{}} — return to default AI
- RESET: {{}} — clear all overrides for team

## Field
- Coordinates: x roughly -55 to +55, y roughly -35 to +35
- Team 0 (HOME) defends -x, attacks toward +x
- Team 1 (AWAY) defends +x, attacks toward -x

## Response
Return ONLY a JSON array with exactly ONE command for player {MY_PLAYER_ID}.
Example: [{{"commandType":"MARK","playerId":{MY_PLAYER_ID},"parameters":{{"target_player_id":4,"tightness":"TIGHT"}},"duration":5}}]
Return ONLY the JSON array, no text before or after."""


# --- Fallback ---

fallback_commands = build_fallback(DEF_CONFIG)


# --- Wire it up ---

agent, mcp_client = create_combined_agent(SYSTEM_PROMPT, MY_PLAYER_ID, POSITION_LABEL, model_id="us.amazon.nova-lite-v1:0")
create_combined_invoke_handler(
    app, agent, mcp_client, MY_PLAYER_ID, POSITION_LABEL, fallback_commands,
    fallback_cfg=DEF_CONFIG,
)

if __name__ == "__main__":
    app.run()
