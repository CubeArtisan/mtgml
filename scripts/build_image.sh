#!/usr/bin/env bash

set -e

rm -r ml_files/latest/* ml_files/testing_tflite
mkdir -p ml_files/latest
cp -r data/maps/int_to_card.json ml_files/latest
cp data/maps/original_to_new_index.json ml_files/latest
cp ml_files/train_prod/* ml_files/latest
python -m mtgml.postprocessing.patch_model
docker buildx build --platform linux/arm64/v8,linux/amd64 --tag $REPOSITORY/mtgml:$TAG . -f .docker/Dockerfile.eval --push
docker buildx build --platform linux/arm64/v8,linux/amd64 --tag $REPOSITORY/mtgml:latest . -f .docker/Dockerfile.eval --push
