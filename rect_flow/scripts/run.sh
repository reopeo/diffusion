#!/bin/bash

datasets=(mnist)

for ds in ${datasets[@]};
do
    config_path=../configs/$ds.yaml
    echo $config_path
    python ../src/train.py --config $config_path
done
