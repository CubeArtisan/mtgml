#!/usr/bin/env bash

set -e

# export SET_TYPES=("mom" "sir" "one" "bro" "dmu" "snc" "neo" "vow" "mid" "afr" "stx")
# Older sets need different processing
# MOM is left out since something weird is causing the card data to be missing kenrith the returned king
export SET_TYPES=("sir" "one" "bro" "dmu" "snc" "neo")
export GS_PATH=gs://cubeartisan/exports/
export DATE=`gsutil ls -lh $GS_PATH\
    | sed 's/^ *//g'\
    | cut -f 6 -d " "\
    | head -n -2\
    | sort\
    | tail -n 1\
    | cut -d '/' -f 5\
    | cut -d '.' -f 1`

rm -rf data/17lands/complete
mkdir -p data/17lands/complete
for set in ${SET_TYPES[@]}
do
    ./scripts/set_ci.sh $set
    cp data/17lands/$set/PremierDraft.csv.json data/17lands/complete/${set}PremierDraft.json
    cp data/17lands/$set/TradDraft.csv.json data/17lands/complete/${set}TradDraft.json
done
./scripts/prod_ci.sh

docker buildx build --platform linux/arm64/v8,linux/amd64 --tag $REPOSITORY/mtgml:$DATE --tag $REPOSITORY/mtgml:latest . -f .docker/Dockerfile --push
