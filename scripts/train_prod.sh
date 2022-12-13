#!/usr/bin/env bash

set -e

require_clean_work_tree () {
    # Update the index
    git update-index -q --ignore-submodules --refresh
    err=0

    # Disallow unstaged changes in the working tree
    if ! git diff-files --quiet --ignore-submodules --
    then
        echo >&2 "cannot $1: you have unstaged changes."
        git diff-files --name-status -r --ignore-submodules -- >&2
        err=1
    fi

    # Disallow uncommitted changes in the index
    if ! git diff-index --cached --quiet HEAD --ignore-submodules --
    then
        echo >&2 "cannot $1: your index contains uncommitted changes."
        git diff-index --cached --name-status -r --ignore-submodules HEAD -- >&2
        err=1
    fi

    if [ $err = 1 ]
    then
        echo >&2 "Please commit or stash them."
        exit 1
    fi
}

require_clean_work_tree
export TAG=${GITHUB_SHA:0:8}

rm -rf ml_files/train_prod
mkdir -p ml_files/train_prod
echo $TAG > git-commit
cp examples/prod.yaml ml_files/train_prod/hyper_config.yaml
python -m mtgml.training.train_combined --name train_prod --epochs 1000 --seed 268459

rm -rf ml_files/latest
mkdir ml_files/latest
cp ml_files/train_prod/* ml_files/latest
cp data/maps/int_to_card.json ml_files/latest
cp data/maps/original_to_new_index.json ml_files/latest

rm -r ml_files/testing_tflite
python -m mtgml.postprocessing.patch_model

export REPOSITORY=ghcr.io/cubeartisan
export REPOSITORY2=gcr.io/cubeartisan

docker-compose -f .docker/docker-compose.yml build
docker-compose -f .docker/docker-compose.yml push
docker tag $REPOSITORY/mtgml:$TAG $REPOSITORY2/mtgml:$TAG
docker push $REPOSITORY2/mtgml:$TAG
