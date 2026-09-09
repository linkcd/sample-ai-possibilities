"""Local test for the DEF1 (Memory + Gateway) agent — tests state summary, parsing, and fallback."""

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "lib"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from test_helpers import mock_agentcore_combined, GAME_STATE, TEAM_ID
os.environ.setdefault("MEMORY_TEAM_MEMORY_ID", "test-memory-id")
os.environ.setdefault("AGENTCORE_GATEWAY_TACTICAL_TOOLS_URL", "https://test-gateway.example.com")
os.environ.setdefault("TEAM_ID", str(TEAM_ID))
mock_agentcore_combined()

from state import summarize_state
from parsing import parse_commands
from main import fallback_commands, MY_PLAYER_ID, POSITION_LABEL


def test_summarize():
    print(f"=== STATE SUMMARY ({POSITION_LABEL}, player {MY_PLAYER_ID}) ===")
    summary = summarize_state(GAME_STATE, TEAM_ID, MY_PLAYER_ID, POSITION_LABEL)
    print(summary)
    print()


def test_fallback():
    print(f"=== FALLBACK ({POSITION_LABEL}) ===")
    cmds = fallback_commands(GAME_STATE, TEAM_ID, MY_PLAYER_ID)
    for c in cmds:
        pid = c.get("playerId")
        tid = c.get("teamId")
        ok = "OK" if pid == MY_PLAYER_ID and tid == TEAM_ID else "WRONG"
        print(f"  [{ok}] P{pid} T{tid}: {c['commandType']} {c.get('parameters', {})}")
    assert all(c["playerId"] == MY_PLAYER_ID for c in cmds), "FAIL: wrong playerId"
    assert all(c["teamId"] == TEAM_ID for c in cmds), "FAIL: wrong teamId"
    print(f"  All {len(cmds)} commands correct")
    print()


def test_fallback_with_ball():
    """Test fallback when DEF has the ball — should PASS to nearest non-GK teammate."""
    print(f"=== FALLBACK WITH BALL ({POSITION_LABEL}) ===")
    state = json.loads(json.dumps(GAME_STATE))
    state["ball"]["possessionAgentId"] = f"agentId_{MY_PLAYER_ID}"
    cmds = fallback_commands(state, TEAM_ID, MY_PLAYER_ID)
    for c in cmds:
        print(f"  P{c['playerId']}: {c['commandType']} {c.get('parameters', {})}")
    assert cmds[0]["commandType"] == "PASS", f"FAIL: expected PASS, got {cmds[0]['commandType']}"
    target = cmds[0]["parameters"]["target_player_id"]
    assert target != 0, "FAIL: DEF should not pass to GK"
    print(f"  Correctly passes to player {target}")
    print()


def test_fallback_opponent_near_goal():
    """Test that DEF marks opponent close to our goal."""
    print(f"=== FALLBACK OPPONENT NEAR GOAL ({POSITION_LABEL}) ===")
    state = json.loads(json.dumps(GAME_STATE))
    state["ball"]["possessionAgentId"] = None
    state["players"][6]["position"] = {"x": -40, "y": 5}  # opp P1 near our goal
    cmds = fallback_commands(state, TEAM_ID, MY_PLAYER_ID)
    for c in cmds:
        print(f"  P{c['playerId']}: {c['commandType']} {c.get('parameters', {})}")
    assert cmds[0]["commandType"] == "MARK", f"FAIL: expected MARK, got {cmds[0]['commandType']}"
    print(f"  Correctly marks dangerous opponent")
    print()


def test_parse():
    print("=== PARSE TESTS ===")
    tests = [
        ('[{"commandType":"MARK","playerId":1,"parameters":{"target_player_id":3,"tightness":"TIGHT"},"duration":5}]', 1),
        ('Here:\n[{"commandType":"PASS","playerId":1,"parameters":{"target_player_id":2,"type":"GROUND"},"duration":0}]\nDone!', 1),
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
    test_fallback_with_ball()
    test_fallback_opponent_near_goal()
    test_parse()
    print("Combined memory+gateway agent local tests passed (no LLM/Memory/Gateway calls).")
    print("Deploy to AgentCore to test with actual Memory + tactical tools integration.")
