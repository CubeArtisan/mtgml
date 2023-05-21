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

if ! [[ -d $DATE ]]
then
    gsutil cp $GS_PATH$DATE.tar.xz $DATE.tar.xz
    tar xJf $DATE.tar.xz
    rm $DATE.tar.xz
fi
cp $DATE/int_to_card.json maps/int_to_card.json
cp $DATE/card_to_int.json maps/card_to_int.json

cd ..
python -m mtgml.preprocessing.find_used data/$DATE/drafts\;data/$DATE/draftlogs\;data/17lands/complete data/$DATE/cubes data/CubeCobra/cubes
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
python -m mtgml.preprocessing.load_picks data/$DATE/drafts\;data/$DATE/draftlogs\;data/17lands/complete

export GITHUB_SHA=`git rev-parse HEAD`
export TYPE=prod

rm -rf ml_files/train_$TYPE
mkdir -p ml_files/train_$TYPE
echo $GITHUB_SHA > ml_files/train_$TYPE/git-commit
# cp examples/$TYPE.pre.yaml ml_files/train_$TYPE/hyper_config.yaml
cp data/maps/original_to_new_index.json ml_files/train_$TYPE/original_to_new_index.json
# python -m mtgml.training.train_combined --name train_$TYPE --epochs 8 --seed 268459 || true
# cp ml_files/train_$TYPE/hyper_config.yaml ml_files/train_$TYPE/hyper_config.pre.yaml
cp examples/$TYPE.yaml ml_files/train_$TYPE/hyper_config.yaml
python -m mtgml.training.train_combined --name train_$TYPE --epochs 32 --seed 16809

export REPOSITORY=ghcr.io/cubeartisan

rm -r ml_files/latest/* ml_files/testing_tflite
mkdir -p ml_files/latest
cp -r data/maps/int_to_card.json ml_files/latest
cp data/maps/original_to_new_index.json ml_files/latest
cp ml_files/train_$TYPE/* ml_files/latest
python -m mtgml.postprocessing.patch_combined $TYPE
docker buildx build --platform linux/arm64/v8,linux/amd64 --tag $REPOSITORY/mtgml:$TYPE-$DATE --tag $REPOSITORY/mtgml:$TYPE-latest . -f .docker/Dockerfile.$TYPE --push
