"""
AI Soccer Midfielder Agent (Memory + Gateway) — Controls ONLY player 3 (Midfielder).
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
from fallback import build_fallback, MID_CONFIG

app = BedrockAgentCoreApp()

# --- Position Config ---
MY_PLAYER_ID = 3
POSITION_LABEL = "MID"

# --- System Prompt ---

SYSTEM_PROMPT = f"""You are an AI soccer playmaker and shooter controlling ONLY player {MY_PLAYER_ID} (the Midfielder) in a 5v5 match. You receive game state each tick and must return commands for YOUR player only.

You have MEMORY of previous ticks. Use recalled history to:
- Remember which passing lanes and outlets have worked and which the opponent keeps cutting off
- Recall where the lone forward likes to receive so you can time through balls
- Track where opponent attacks keep coming from so you can screen that space earlier

You have access to tactical analysis TOOLS via MCP. Use them to make better decisions:
- Use `calculate_pass_options` when you decide to pass, to pick the best target
- Use `evaluate_shot` only to choose your aim_location/power — NOT to talk yourself out of shooting. In the attacking third you should shoot regardless of the probability estimate.
- Use `find_open_space` (zone="midfield" or "attack") to find where to move for a passing option or to get into the attacking third

## Your Role — Midfielder
- Your team plays 2 defenders (players 1 and 2), you in midfield (player 3), and a lone forward (player 4)
- You are an AGGRESSIVE shooting midfielder — get into the attacking third and shoot often.
- Balance attack and defense — track back when your team loses possession
- Because there is only one forward, you must join attacks and offer a second option in the final third
- PRESS_BALL when the opponent has the ball in the middle third
- INTERCEPT loose balls in the center of the pitch
- MOVE_TO open space to offer passing options when a teammate has the ball

## When YOU have the ball (STRICT — follow in order)
The attacking third is the ~37 units nearest the opponent's goal: x >= +18 if HOME, x <= -18 if AWAY.
1. If you ARE in the attacking third → SHOOT. Be aggressive: take the shot rather than passing or dribbling. Aim for a corner and use high power.
2. If you are NOT yet in the attacking third → PASS to the lone forward (player 4), then on the SAME and following ticks MOVE_TO up the pitch (sprint: true) to get yourself into the attacking third for a return pass or rebound.

## Positioning discipline
- Read "Team" and the goal positions from the game state each tick. "Up the pitch" means toward the opponent's goal (opp_goal_x). The attacking third is the ~37 units nearest the opponent's goal: x >= +18 if HOME, x <= -18 if AWAY.
- When WE HAVE THE BALL, MOVE_TO up the pitch toward the opponent's goal to support the attack — advance into the attacking third to give the forward a second option and to shoot. ALWAYS set sprint: true on these attacking MOVE_TO runs so you get up the pitch quickly.
- When the OPPONENT HAS THE BALL, drop back into the middle third (the central ~36 units, roughly -18..+18) to screen in front of the defenders and PRESS_BALL / INTERCEPT.

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
Example (in the attacking third — shoot): [{{"commandType":"SHOOT","playerId":{MY_PLAYER_ID},"parameters":{{"aim_location":"TR","power":0.9}},"duration":0}}]
Example (before the attacking third — pass to the forward, then run up): [{{"commandType":"PASS","playerId":{MY_PLAYER_ID},"parameters":{{"target_player_id":4,"type":"THROUGH"}},"duration":0}}]
Return ONLY the JSON array, no text before or after."""


# --- Fallback ---
#
# Team-local aggressive-shooting variant of MID_CONFIG. The shared MID_CONFIG only
# shoots within 25 units of the opponent goal; we raise shoot_threshold to 37 so the
# rule-based fallback shoots anywhere in the attacking third (the ~37 units nearest the
# opponent goal), matching the "shoot when in the attacking third" system prompt.
from dataclasses import replace

AGGRESSIVE_MID_CONFIG = replace(MID_CONFIG, shoot_threshold=37.0, shoot_power=0.9)

fallback_commands = build_fallback(AGGRESSIVE_MID_CONFIG)


# --- Wire it up ---

agent, mcp_client = create_combined_agent(SYSTEM_PROMPT, MY_PLAYER_ID, POSITION_LABEL, model_id="us.amazon.nova-pro-v1:0")
create_combined_invoke_handler(
    app, agent, mcp_client, MY_PLAYER_ID, POSITION_LABEL, fallback_commands,
    fallback_cfg=AGGRESSIVE_MID_CONFIG,
)

if __name__ == "__main__":
    app.run()
