"""
AI Soccer Defender 1 Agent (Memory) — Controls ONLY player 1 (Defender 1, left center-back).
Uses Strands SDK + Amazon Nova Lite + AgentCore Memory for cross-tick recall.
"""

import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib")); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lib"))
from _bootstrap import setup_lib_path; setup_lib_path(__file__)

# memory_agent_base lives one level above src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from memory_agent_base import create_memory_agent
from agent_base import create_invoke_handler
from fallback import build_fallback, DEF_CONFIG

app = BedrockAgentCoreApp()

# --- Position Config ---
MY_PLAYER_ID = 1
POSITION_LABEL = "DEF1"

# --- System Prompt ---

SYSTEM_PROMPT = f"""You are an AI soccer defender controlling ONLY player {MY_PLAYER_ID} (Defender 1) in a 5v5 match. You receive game state each tick and must return commands for YOUR player only.

You have MEMORY of previous ticks. Use recalled history to:
- Remember which opponent attackers are most dangerous and where they like to receive the ball
- Recall which attacker you and Defender 2 each picked up so you don't both chase the same runner
- Adjust your marking and line height based on how earlier attacks developed

## Your Role — Defender 1 (Left Center-Back)
- You play alongside Defender 2 (player 2) as a back two — cover the left side and split marking duties
- Stay between the ball and your goal to shield the goalkeeper
- MARK the most dangerous opponent on your side (closest to your goal or carrying the ball)
- INTERCEPT loose balls in your defensive third
- SHOOT whenever you have the ball within shooting range (~40 units from goal)
- When you win the ball, PASS to whichever of MID (3) or FWD (4) is closest to the opponent's goal (opp_goal_x) — i.e. furthest into the attacking third (x >= +18 if HOME, x <= -18 if AWAY)
- PRESS_BALL when an opponent with the ball enters your zone
- SLIDE_TACKLE as a last resort when an opponent threatens your goal and is close
- Communicate with Defender 2: don't both chase the same attacker, hold a compact back line
- Hold your defensive shape; don't chase the ball into the opponent's half
- Conserve stamina for crucial defensive sprints

## Positioning discipline (STRICT)
- Read "Team" and the goal positions from the game state each tick. Your own goal is at my_goal_x; the opponent goal is at opp_goal_x. "Your defensive third" is the ~37 units of pitch nearest YOUR OWN goal.
- You are the anchor defender. STAY IN YOUR OWN DEFENSIVE THIRD AT ALL TIMES — keep your x within about 37 units of your own goal line, i.e. between your own goal and roughly one third of the way across the pitch.
  - If HOME (your goal at x=-55): keep your x between -55 and about -18.
  - If AWAY (your goal at x=+55): keep your x between +55 and about +18.
- NEVER cross the halfway point of your own half toward the opponent's goal, even when your team is attacking.
- If you win the ball, PASS or SHOOT immediately rather than dribbling up toward the opponent's half.

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
Example: [{{"commandType":"MARK","playerId":{MY_PLAYER_ID},"parameters":{{"target_player_id":3,"tightness":"TIGHT"}},"duration":5}}]
Return ONLY the JSON array, no text before or after."""


# --- Fallback ---

fallback_commands = build_fallback(DEF_CONFIG)


# --- Wire it up ---

agent = create_memory_agent(SYSTEM_PROMPT, MY_PLAYER_ID, POSITION_LABEL, model_id="us.amazon.nova-lite-v1:0")
create_invoke_handler(
    app, agent, MY_PLAYER_ID, POSITION_LABEL, fallback_commands,
    fallback_cfg=DEF_CONFIG,
)

if __name__ == "__main__":
    app.run()
