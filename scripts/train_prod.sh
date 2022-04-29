#!/usr/bin/env bash

export GS_PATH=gs://cubeartisan/exports/
# export DATE=`gsutil ls -lh $GS_PATH\
#     | sed 's/^ *//g'\
#     | cut -f 6 -d " "\
#     | head -n -2\
#     | sort\
#     | tail -n 1\
#     | cut -d '/' -f 5\
#     | cut -d '.' -f 1`
export DATE=20220321

# mkdir -p data/CubeCobra/drafts data/CubeCobra/decks data/CubeCobra/cubes data/maps
# cd data
# rm -rf $DATE maps/int_to_card.json maps/card_to_int.json

# gsutil cp $GS_PATH$DATE.tar.xz $DATE.tar.xz
# tar xJf $DATE.tar.xz
# ln -s ../$DATE/int_to_card.json maps/int_to_card.json
# ln -s ../$DATE/card_to_int.json maps/card_to_int.json

# cd ..

# python -m mtgml.preprocessing.load_decks data/$DATE/decks data/CubeCobra/decks
# python -m mtgml.preprocessing.load_cubes data/$DATE/cubes data/CubeCobra/cubes
# python -m mtgml.preprocessing.load_picks data/$DATE/drafts\;data/$DATE/draftlogs data/CubeCobra/drafts

rm -rf ml_files/train_prod
mkdir -p ml_files/train_prod
cp examples/*_hyper_config.yaml ml_files/train_prod
timeout 1h sh -c 'for i in {15801..15900}; do python -m mtgml.training.train_cards_combined --name train_prod --time 60 --epochs 1000 --seed $i; done'
timeout 3h sh -c 'for i in {25801..25900}; do python -m mtgml.training.train_cards_combined --name train_prod --time 120 --epochs 1000 --seed $i --fine_tuning; done'
timeout 1h sh -c 'for i in {35801..35900}; do python -m mtgml.training.train_draftbots --name train_prod --time 60 --epochs 1000 --seed $i; done'
timeout 1h sh -c 'for i in {45801..45900}; do python -m mtgml.training.train_recommender --name train_prod --time 60 --epochs 1000 --seed $i; done'

rm -rf ml_files/latest
mkdir ml_files/latest
cp ml_files/train_prod/* ml_files/latest
cp data/maps/int_to_card.json ml_files/latest

export REPOSITORY=ghcr.io/cubeartisan
export TAG=$DATE

docker-compose -f .docker/docker-compose.yml build && docker-compose -f .docker/docker-compose.yml up

