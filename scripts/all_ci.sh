#!/usr/bin/env bash

set -e

export SET_TYPES=("one" "bro" "dmu" "snc" "neo" "vow" "mid" "afr" "stx")

for set in ${SET_TYPES[@]}
do
    ./scripts/set_ci.sh $set
done
rm -rf data/17lands/complete
mkdir -p data/17lands/complete
cp data/17lands/*/*.json data/17lands/complete
./scripts/prod_ci.sh
