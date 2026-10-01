#!/bin/bash
# usage: runmany.sh jobsfile ; each line: outname args...
cd /home/claude/work
export COSE_THREADS=1
cat $1 | while read name args; do
  if [ ! -f runs/$name.json ]; then echo "python3 cose_sim.py $args --out runs/$name.json > logs/$name.log 2>&1"; fi
done | xargs -P 2 -I{} bash -c "{}"
