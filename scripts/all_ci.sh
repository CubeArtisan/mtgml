#!/usr/bin/env bash

set -e

# export SET_TYPES=("one" "bro" "dmu" "snc" "neo" "vow" "mid" "afr" "stx")
# Older sets need different processing
export SET_TYPES=("one" "bro" "dmu" "snc" "neo")

rm -rf data/17lands/complete
mkdir -p data/17lands/complete
for set in ${SET_TYPES[@]}
do
    ./scripts/set_ci.sh $set
    cp data/17lands/$set/PremierDraft.csv.json data/17lands/complete/${set}PremierDraft.json
    cp data/17lands/$set/TradDraft.csv.json data/17lands/complete/${set}TradDraft.json
done
./scripts/prod_ci.sh
