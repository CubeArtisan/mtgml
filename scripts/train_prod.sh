#!/usr/bin/env bash

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

docker-compose -f .docker/docker-compose.yml build
docker-compose -f .docker/docker-compose.yml push
