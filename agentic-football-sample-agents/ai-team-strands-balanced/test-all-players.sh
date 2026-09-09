#!/bin/bash
for pos in {"gk","def","mid","fwd1","fwd2"}; do python3 "ai-${pos}/test_local.py" --llm; done;
# end of script
