#!/usr/bin/env bash

set -e

export SET_TYPES=("ltr" "mom" "sir" "one" "bro" "dmu" "snc" "neo" "vow" "mid" "stx")
# HBG seems to have issues with looking up card names, needs further investigation
# KHM doesn't have any draft data so a set specific model doesn't make sense currently
# AFR seems to have file corruption issues though that might be local
# export SET_TYPES=("ltr" "mom" "sir" "one" "bro" "dmu" "hbg" "snc" "neo" "vow" "mid" "afr" "stx" "khm")
export GS_PATH=gs://cubeartisan/exports/
export DATE=`gsutil ls -lh $GS_PATH\
    | sed 's/^ *//g'\
    | cut -f 6 -d " "\
    | head -n -2\
    | sort\
    | tail -n 1\
    | cut -d '/' -f 5\
    | cut -d '.' -f 1`

rm -rf data/17lands_complete
mkdir -p data/17lands_complete/drafts data/17lands_complete/decks
for set in ${SET_TYPES[@]}
do
    ./scripts/set_ci.sh $set
    cd data/17lands/$set/draft
    for name in *.json
    do
      cp $name ../../../17lands_complete/drafts/${set}_${name}
    done
    cd ../game
    for name in *.json
    do
      cp $name ../../../17lands_complete/decks/${set}_${name}
    done
    cd ../../../../
done
./scripts/prod_ci.sh

docker buildx build --platform linux/arm64/v8,linux/amd64 --tag $REPOSITORY/mtgml:$DATE --tag $REPOSITORY/mtgml:latest . -f .docker/Dockerfile --push
