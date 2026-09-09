#!/bin/bash
for agent in "ai-gk" "ai-def1" "ai-mid" "ai-def2" "ai-fwd1"; do python3 "${agent}/test_local.py"; done;
# end of script
