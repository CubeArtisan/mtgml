#!/usr/bin/env bash

set -e

mkdir -p data/CubeCobra/drafts data/CubeCobra/decks data/CubeCobra/cubes data/CubeCobra/picks data/maps
cd data
rm -rf maps/int_to_card.json maps/card_to_int.json

if aws s3 ls s3://cubecobra/ > /dev/null; then
  aws s3 sync s3://cubecobra/decks CubeCobra/decks
  aws s3 sync s3://cubecobra/picks CubeCobra/picks
  aws s3 cp s3://cubecobra/cubes.json CubeCobra/cubes/cubes.json
  aws s3 cp s3://cubecobra/indexToOracleMap.json CubeCobra/indexToOracleMap.json
else
  echo "You don't have access to the CubeCobra aws bucket. Talk to Dekkaru to get access." 1>2
fi

export GS_PATH=gs://cubeartisan/exports/
export DATE=`gsutil ls -lh $GS_PATH\
    | sed 's/^ *//g'\
    | cut -f 6 -d " "\
    | head -n -2\
    | sort\
    | tail -n 1\
    | cut -d '/' -f 5\
    | cut -d '.' -f 1`

gsutil cp $GS_PATH$DATE.tar.xz $DATE.tar.xz
rm -rf $DATE
tar xJf $DATE.tar.xz
rm $DATE.tar.xz
cp $DATE/int_to_card.json maps/int_to_card.json
cp $DATE/card_to_int.json maps/card_to_int.json

cd ..
python -m mtgml.preprocessing.find_used data/$DATE/drafts\;data/$DATE/draftlogs data/$DATE/cubes data/CubeCobra/cubes
if [[ -d data/CubeCobra/decks ]]; then
    python -m mtgml.preprocessing.load_decks data/$DATE/decks data/CubeCobra/decks
else
    python -m mtgml.preprocessing.load_decks data/$DATE/decks
fi
if [[ -d data/CubeCobra/cubes ]]; then
    python -m mtgml.preprocessing.load_cubes data/$DATE/cubes data/CubeCobra/cubes
else
    python -m mtgml.preprocessing.load_cubes data/$DATE/cubes
fi
python -m mtgml.preprocessing.load_picks data/$DATE/drafts\;data/$DATE/draftlogs
