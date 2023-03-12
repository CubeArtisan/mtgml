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

mkdir -p data/maps
cd data
rm -rf maps/int_to_card.json maps/card_to_int.json maps/original_to_new_index.json

# Still use this for latest card data
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

mkdir -p 17lands/$1
cd 17lands/$1

export FILENAME_BASE=draft_data_public.${1^^}
export DRAFT_TYPES=("Trad" "Premier")
for DRAFT_TYPE in ${DRAFT_TYPES[@]}
do
    export FILENAME=${DRAFT_TYPE}Draft.csv
    if [[ ! -f ${FILENAME}.json ]]
    then
        curl https://17lands-public.s3.amazonaws.com/analysis_data/draft_data/$FILENAME_BASE.$FILENAME.gz --output $FILENAME.gz \
            && gzip -fd ${FILENAME}.gz \
            || true
    fi
done

cd ../../../

for DRAFT_TYPE in ${DRAFT_TYPES[@]}
do
    if [[ -f data/17lands/$1/${DRAFT_TYPE}Draft.csv ]]
    then
        python -m mtgml.preprocessing.17lands_to_json data/17lands/$1/${DRAFT_TYPE}Draft.csv
        rm data/17lands/$1/${DRAFT_TYPE}Draft.csv
    fi
done
python -m mtgml.preprocessing.find_used data/17lands/$1 nonexistant nonexistant
python -m mtgml.preprocessing.load_picks data/17lands/$1

export GITHUB_SHA=`git rev-parse HEAD`
export TYPE=$1

rm -rf ml_files/train_$TYPE
mkdir -p ml_files/train_$TYPE
echo $GITHUB_SHA > ml_files/train_$TYPE/git-commit
cp data/maps/original_to_new_index.json ml_files/train_$TYPE/original_to_new_index.json
cp examples/draftbots.set.yaml ml_files/train_$TYPE/hyper_config.yaml
python -m mtgml.training.train_draftbots --name train_$TYPE --epochs 64 --seed 16809

export REPOSITORY=ghcr.io/cubeartisan

rm -r ml_files/latest/* ml_files/testing_tflite
mkdir -p ml_files/latest
cp -r data/maps/int_to_card.json ml_files/latest
cp data/maps/original_to_new_index.json ml_files/latest
cp ml_files/train_$TYPE/* ml_files/latest
python -m mtgml.postprocessing.patch_draftbots
docker buildx build --platform linux/arm64/v8,linux/amd64 --tag $REPOSITORY/mtgml:$TYPE-$DATE --tag $REPOSITORY/mtgml:$TYPE-latest . -f .docker/Dockerfile.eval --push
