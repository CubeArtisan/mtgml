#!/usr/bin/env bash

set -e

mkdir -p data/CubeCobra/drafts data/CubeCobra/decks data/CubeCobra/cubes data/maps
cd data
rm -rf maps/int_to_card.json maps/card_to_int.json

# if aws s3 ls s3://cubecobra/ > /dev/null; then
#   mkdir -p CubeCobra
#   aws s3 sync s3://cubecobra/draft_picks CubeCobra/drafts
#   aws s3 sync s3://cubecobra/deck_exports CubeCobra/decks
#   aws s3 sync s3://cubecobra/cube_exports CubeCobra/cubes
#   parallel sed "'s/[$#]:\"/\":\"/g'" -i {} ::: CubeCobra/{decks,drafts}/*.json
#   parallel sed "'s/\`/\"/g'" -i {} ::: CubeCobra/{decks,drafts}/*.json
# else
#   echo "You don't have access to the CubeCobra aws bucket. Talk to Dekkaru to get access." 1>2
# fi

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
python ../mtgml/preprocessing/find_used.py data/$DATE/drafts\;data/$DATE/draftlogs data/CubeCobra/drafts data/$DATE/cubes data/CubeCobra/cubes
if [[ -d data/CubeCobra/decks ]]; then
    python ../mtgml/preprocessing/load_decks.py data/$DATE/decks data/CubeCobra/decks
else
    python ../mtgml/preprocessing/load_decks.py data/$DATE/decks
fi
if [[ -d data/CubeCobra/cubes ]]; then
    python ../mtgml/preprocessing/load_cubes.py data/$DATE/cubes data/CubeCobra/cubes
else
    python ../mtgml/preprocessing/load_cubes.py data/$DATE/cubes
fi
if [[ -d data/CubeCobra/drafts ]]; then
    python ../mtgml.preprocessing/load_picks.py data/$DATE/drafts\;data/$DATE/draftlogs data/CubeCobra/drafts
else
    python ../mtgml.preprocessing/load_picks.py data/$DATE/drafts\;data/$DATE/draftlogs
fi
