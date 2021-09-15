#!/bin/sh
set -e
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'
DATA_DIR='/home/kaggler/Documents/data'
CODE_DIR='~/Documents/brain-tumor-classification-'${USER}

echo "${GREEN}[1/3] COPYING FILES...${NC}"
eval $(ssh-agent)
rsync -avz --exclude='.git' --filter=':- .gitignore' . kagglepc:${CODE_DIR}

echo "${GREEN}[2/3] START TRAINING...${NC}"
FREE_MEM=$(ssh kagglepc nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
if [ $FREE_MEM -lt 1000 ]; then
    echo "${RED}Not enough GPU memory!${NC}"
    exit
fi
ssh kagglepc "cd ${CODE_DIR}; \
    docker run --gpus all -u \$(id -u):\$(id -g) --name tf-gpu-${USER} \
    --rm -v \$(pwd):/work -v ${DATA_DIR}:/work/data tf-gpu python train.py"

echo "${GREEN}[3/3] GET RESULTS...${NC}"
scp -r kagglepc:${CODE_DIR}/exps/* exps/
ssh kagglepc "rm -rf ${CODE_DIR}/exps/*"