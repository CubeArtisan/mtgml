#!/usr/bin/env bash

export GS_PATH=gs://cubeartisan/exports/
export DATE=`gsutil ls -lh $GS_PATH\
    | sed 's/^ *//g'\
    | cut -f 6 -d " "\
    | head -n -2\
    | sort\
    | tail -n 1\
    | cut -d '/' -f 5\
    | cut -d '.' -f 1`

mkdir -p data/CubeCobra/drafts data/CubeCobra/decks data/CubeCobra/cubes data/maps
cd data
rm -r $DATE maps/int_to_card.json maps/card_to_int.json

gsutil cp $GS_PATH$DATE.tar.xz $DATE.tar.xz
tar xJf $DATE.tar.xz
ln -s ../$DATE/int_to_card.json maps/int_to_card.json
ln -s ../$DATE/card_to_int.json maps/card_to_int.json

cd ..

python -m mtgml.preprocessing.load_decks data/$DATE/decks data/CubeCobra/decks
python -m mtgml.preprocessing.load_cubes data/$DATE/cubes data/CubeCobra/cubes
python -m mtgml.preprocessing.load_picks data/$DATE/drafts\;data/$DATE/draftlogs data/CubeCobra/drafts

rm -rf ml_files/train_prod
mkdir -p ml_files/train_prod
cp examples/prod.yaml ml_files/train_prod
python -m mtgml.training.train_combined --name train_prod --epochs 1000 --seed 268459 --time 360

rm -rf ml_files/latest
mkdir ml_files/latest
cp ml_files/train_prod/* ml_files/latest
cp data/maps/int_to_card.json ml_files/latest

export REPOSITORY=ghcr.io/cubeartisan
export TAG=$DATE

docker-compose -f .docker/docker-compose.yml push

