#!/bin/bash
set -e
cd "$(dirname "$0")/../proof"
cat > /tmp/tsimd_lean_axioms.lean <<'EOF'
import Tokensim.Bpe
#print axioms bpeFuel_append
#print axioms decode_bpeFuel
#print axioms bpeFuel_fuel_independent
#print axioms bpeFuel_normal_form
#print axioms bpeFuel_stable
#print axioms decode_bpeEncode
#print axioms bpeEncode_append
#print axioms decode_append
#print axioms decodeIds_encodeIds
#print axioms minRank_append
#print axioms mergeFirst_append_left
#print axioms mergeFirst_append_right
#print axioms mergeFirst_decode
#print axioms mergeFirst_length_lt
#print axioms decode_last_suffix
#print axioms decode_head_prefix
EOF
lake env lean /tmp/tsimd_lean_axioms.lean
