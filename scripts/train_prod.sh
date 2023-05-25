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
export GITHUB_SHA=`git rev-parse HEAD`
export TAG=${GITHUB_SHA:0:8}

rm -rf ml_files/train_prod
mkdir -p ml_files/train_prod
echo $GITHUB_SHA > ml_files/train_prod/git-commit
cp examples/prod.yaml ml_files/train_prod/hyper_config.yaml
python -m mtgml.training.train_combined --name train_prod --epochs 32 --seed 16809 --log-dir logs/fit/$TYPE-$TAG
