#!/bin/bash
set -eE
OUT_DIR='kaggle_dataset'
trap "rm -rf $OUT_DIR" ERR
CUR_BRANCH=$(git rev-parse --abbrev-ref HEAD)
trap "git checkout -q $CUR_BRANCH ." EXIT

# Copy git tracked files only
if [ -z "$1" ]; then
    TARGET_HASH=$CUR_BRANCH
else
    TARGET_HASH=$(git log --all --format="%H" --grep="${1}" -n 1)
fi
git log -n 1 $TARGET_HASH
read -p "Create kaggle dataset from the commit above? ([y]/n): " -n 1 -r
if [[ $REPLY =~ ^[Nn]$ ]]; then
    echo -e "\nExiting..."
    exit 1
fi
git checkout -q $TARGET_HASH .
rm -rf $OUT_DIR
mkdir -p $OUT_DIR/src
touch $OUT_DIR/src/__init__.py
for f in $(git ls-files); do
    rsync -R $f $OUT_DIR/src
done

# Add result from experiment directory if exist
COMMIT_MSG=( $(git log --format="%B" -n 1 $TARGET_HASH) )
IFS=- read COMMIT_PREFIX EXP_NUM _ <<< ${COMMIT_MSG[0]}
if [ "$COMMIT_PREFIX" = "EXP" ] && [ -d exps/$EXP_NUM* ]; then
    cp -r exps/$EXP_NUM*/* $OUT_DIR
fi
