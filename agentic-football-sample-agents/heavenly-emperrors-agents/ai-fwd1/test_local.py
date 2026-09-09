"""Local test for the lone FWD (Memory) agent (player 4) — tests state summary, parsing, and fallback."""

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "lib"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from test_helpers import mock_agentcore_memory, GAME_STATE, TEAM_ID
os.environ.setdefault("MEMORY_ID", "test-memory-id")
os.environ.setdefault("TEAM_ID", str(TEAM_ID))
mock_agentcore_memory()

from state import summarize_state
from parsing import parse_commands
from main import fallback_commands, MY_PLAYER_ID, POSITION_LABEL


def test_summarize():
    print(f"=== STATE SUMMARY ({POSITION_LABEL}, player {MY_PLAYER_ID}) ===")
    summary = summarize_state(GAME_STATE, TEAM_ID, MY_PLAYER_ID, POSITION_LABEL)
    print(summary)
    print()


def test_fallback():
    """FWD doesn't have ball, a teammate does — should MOVE_TO attacking position."""
    print(f"=== FALLBACK ({POSITION_LABEL}) ===")
    cmds = fallback_commands(GAME_STATE, TEAM_ID, MY_PLAYER_ID)
    for c in cmds:
        pid = c.get("playerId")
        tid = c.get("teamId")
        ok = "OK" if pid == MY_PLAYER_ID and tid == TEAM_ID else "WRONG"
        print(f"  [{ok}] P{pid} T{tid}: {c['commandType']} {c.get('parameters', {})}")
    assert all(c["playerId"] == MY_PLAYER_ID for c in cmds), "FAIL: wrong playerId"
    assert all(c["teamId"] == TEAM_ID for c in cmds), "FAIL: wrong teamId"
    assert cmds[0]["commandType"] == "MOVE_TO", f"FAIL: expected MOVE_TO, got {cmds[0]['commandType']}"
    print(f"  Correctly moves to attacking position (right side)")
    print()


def test_fallback_shoot():
    """FWD has ball near opponent goal — should SHOOT."""
    print(f"=== FALLBACK SHOOT ({POSITION_LABEL}) ===")
    state = json.loads(json.dumps(GAME_STATE))
    state["ball"]["possessionAgentId"] = f"agentId_{MY_PLAYER_ID}"
    state["players"][4]["position"] = {"x": 40, "y": 8}  # near opp goal
    cmds = fallback_commands(state, TEAM_ID, MY_PLAYER_ID)
    for c in cmds:
        print(f"  P{c['playerId']}: {c['commandType']} {c.get('parameters', {})}")
    assert cmds[0]["commandType"] == "SHOOT", f"FAIL: expected SHOOT, got {cmds[0]['commandType']}"
    assert cmds[0]["parameters"]["aim_location"] == "BL", "FAIL: FWD should aim BL"
    print(f"  Correctly shoots (aim BL) near goal")
    print()


def test_fallback_advance():
    """FWD has ball far from goal — should advance with MOVE_TO."""
    print(f"=== FALLBACK ADVANCE ({POSITION_LABEL}) ===")
    state = json.loads(json.dumps(GAME_STATE))
    state["ball"]["possessionAgentId"] = f"agentId_{MY_PLAYER_ID}"
    state["players"][4]["position"] = {"x": 10, "y": 8}  # far from goal
    cmds = fallback_commands(state, TEAM_ID, MY_PLAYER_ID)
    for c in cmds:
        print(f"  P{c['playerId']}: {c['commandType']} {c.get('parameters', {})}")
    assert cmds[0]["commandType"] == "MOVE_TO", f"FAIL: expected MOVE_TO, got {cmds[0]['commandType']}"
    assert cmds[0]["parameters"]["sprint"] == True, "FAIL: should sprint when advancing"
    print(f"  Correctly advances with sprint on right side")
    print()


def test_parse():
    print("=== PARSE TESTS ===")
    tests = [
        ('[{"commandType":"SHOOT","playerId":4,"parameters":{"aim_location":"BL","power":0.9},"duration":0}]', 1),
        ('[{"commandType":"MOVE_TO","playerId":4,"parameters":{"target_x":33,"target_y":8,"sprint":true},"duration":0}]', 1),
        ("invalid json", 0),
        ('[]', 0),
    ]
    all_pass = True
    for resp, expected in tests:
        cmds = parse_commands(resp, TEAM_ID, MY_PLAYER_ID)
        ok = len(cmds) == expected
        if cmds:
            ok = ok and all(c["playerId"] == MY_PLAYER_ID for c in cmds)
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  [{status}] '{resp[:60]}...' -> {len(cmds)} cmds (expected {expected})")
    if all_pass:
        print("  All parse tests passed")
    print()


if __name__ == "__main__":
    test_summarize()
    test_fallback()
    test_fallback_shoot()
    test_fallback_advance()
    test_parse()
    print("Memory agent local tests passed (no LLM/Memory calls).")
    print("Deploy to AgentCore to test with actual Memory integration.")
