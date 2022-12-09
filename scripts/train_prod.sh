#!/usr/bin/env bash

set -e

rm -rf ml_files/train_prod
mkdir -p ml_files/train_prod
cp examples/prod.yaml ml_files/train_prod/hyper_config.yaml
python -m mtgml.training.train_combined --name train_prod --epochs 1000 --seed 268459 --time 720

rm -rf ml_files/latest
mkdir ml_files/latest
cp ml_files/train_prod/* ml_files/latest
cp data/maps/int_to_card.json ml_files/latest

rm -r ml_files/testing_tflite
mkdir ml_files/testing_tflite
cp ml_files/latest/int_to_card.json ml_files/testing_tflite
python -m mtgml.postprocessing.patch_model

export REPOSITORY=ghcr.io/cubeartisan
export REPOSITORY2=gcr.io/cubeartisan

docker-compose -f .docker/docker-compose.yml build
docker-compose -f .docker/docker-compose.yml push
docker tag $REPOSITORY/mtgml:$TAG $REPOSITORY2/mtgml:$TAG
docker push $REPOSITORY2
