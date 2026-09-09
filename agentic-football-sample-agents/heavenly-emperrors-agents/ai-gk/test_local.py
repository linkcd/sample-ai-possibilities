"""Local test for the GK (Memory + Gateway, Solid Defensive) agent — tests state summary, parsing, and fallback."""

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
    assert all(c["playerId"] == MY_PLAYER_ID for c in cmds), "FAIL: wrong playerId in fallback"
    assert all(c["teamId"] == TEAM_ID for c in cmds), "FAIL: wrong teamId in fallback"
    print(f"  All {len(cmds)} commands have correct playerId={MY_PLAYER_ID} and teamId={TEAM_ID}")
    print()


def test_fallback_with_ball():
    """When the GK has the ball, distribute to whichever of MID (3) / FWD (4) is furthest up."""
    print(f"=== FALLBACK WITH BALL ({POSITION_LABEL}) ===")
    state = json.loads(json.dumps(GAME_STATE))
    state["ball"]["possessionAgentId"] = f"agentId_{MY_PLAYER_ID}"
    cmds = fallback_commands(state, TEAM_ID, MY_PLAYER_ID)
    for c in cmds:
        print(f"  P{c['playerId']}: {c['commandType']} {c.get('parameters', {})}")
    assert cmds[0]["commandType"] == "GK_DISTRIBUTE", f"FAIL: expected GK_DISTRIBUTE, got {cmds[0]['commandType']}"
    # Sample state: P3 at x=14, P4 at x=20 (HOME attacks +x), so the forward (4) is furthest up.
    target = cmds[0]["parameters"]["target_player_id"]
    assert target == 4, f"FAIL: expected distribute to furthest-up player 4, got {target}"
    print(f"  Correctly distributes to furthest-up player {target} via {cmds[0]['parameters'].get('method')}")
    print()


def test_fallback_with_ball_mid_furthest():
    """If the midfielder is furthest up, distribution should target the midfielder (3)."""
    print(f"=== FALLBACK WITH BALL — MID FURTHEST ({POSITION_LABEL}) ===")
    state = json.loads(json.dumps(GAME_STATE))
    state["ball"]["possessionAgentId"] = f"agentId_{MY_PLAYER_ID}"
    state["players"][3]["position"] = {"x": 45, "y": 0}   # MID (id 3) pushed high
    state["players"][4]["position"] = {"x": 5, "y": 10}   # FWD (id 4) dropped deep
    cmds = fallback_commands(state, TEAM_ID, MY_PLAYER_ID)
    for c in cmds:
        print(f"  P{c['playerId']}: {c['commandType']} {c.get('parameters', {})}")
    target = cmds[0]["parameters"]["target_player_id"]
    assert target == 3, f"FAIL: expected distribute to furthest-up player 3, got {target}"
    print(f"  Correctly distributes to furthest-up player {target}")
    print()


def test_fallback_positioning():
    """When an opponent has the ball, the keeper holds a deep line near its own goal."""
    print(f"=== FALLBACK POSITIONING ({POSITION_LABEL}) ===")
    state = json.loads(json.dumps(GAME_STATE))
    state["ball"]["possessionAgentId"] = "agentId_6"  # an opponent has the ball
    cmds = fallback_commands(state, TEAM_ID, MY_PLAYER_ID)
    for c in cmds:
        print(f"  P{c['playerId']}: {c['commandType']} {c.get('parameters', {})}")
    assert cmds[0]["commandType"] == "MOVE_TO", f"FAIL: expected MOVE_TO, got {cmds[0]['commandType']}"
    # HOME goal is at x=-55; the keeper should stay deep (well into its own half).
    assert cmds[0]["parameters"]["target_x"] < -40, \
        f"FAIL: keeper should hold a deep line, got target_x={cmds[0]['parameters']['target_x']}"
    assert cmds[0]["parameters"]["sprint"] is False, "FAIL: keeper should not sprint to hold position"
    print(f"  Correctly holds a deep line at x={cmds[0]['parameters']['target_x']}")
    print()


def test_parse():
    print("=== PARSE TESTS ===")
    tests = [
        ('[{"commandType":"GK_DISTRIBUTE","playerId":0,"parameters":{"target_player_id":1,"method":"THROW"},"duration":0}]', 1),
        ('Here:\n[{"commandType":"MOVE_TO","playerId":0,"parameters":{"target_x":-49,"target_y":2,"sprint":false},"duration":0}]\nDone!', 1),
        ('{"commandType":"SET_STANCE","playerId":0,"parameters":{"stance":2},"duration":0}', 1),
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
        print("  All parse tests passed, playerId correctly forced")
    print()


if __name__ == "__main__":
    test_summarize()
    test_fallback()
    test_fallback_with_ball()
    test_fallback_with_ball_mid_furthest()
    test_fallback_positioning()
    test_parse()
    print("Combined memory+gateway agent local tests passed (no LLM/Memory/Gateway calls).")
    print("Deploy to AgentCore to test with actual Memory + tactical tools integration.")
