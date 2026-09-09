#!/bin/bash
# Run the local tests for all five heavenly-emperrors agents.
# These tests are fully mocked (no AWS/LLM/Memory calls) — deploy to AgentCore
# to test the real Memory integration.

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Pick a Python that has the project dependencies (strands, bedrock-agentcore).
# Preference order: an already-activated venv, then the workspace .venv, then python3.
if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
  PYTHON="$VIRTUAL_ENV/bin/python"
elif [ -x "$SCRIPT_DIR/../../.venv/bin/python" ]; then
  PYTHON="$SCRIPT_DIR/../../.venv/bin/python"
else
  PYTHON="python3"
fi

echo "Using Python: $PYTHON"
echo ""

FAILED=()
for agent in "ai-gk" "ai-def1" "ai-def2" "ai-mid" "ai-fwd1"; do
  echo "========== $agent =========="
  if ! "$PYTHON" "$SCRIPT_DIR/$agent/test_local.py"; then
    FAILED+=("$agent")
  fi
  echo ""
done

if [ ${#FAILED[@]} -gt 0 ]; then
  echo "FAILED: ${FAILED[*]}"
  exit 1
fi
echo "All agents passed."
# end of script
