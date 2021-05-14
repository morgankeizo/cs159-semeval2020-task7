#!/usr/bin/env bash

plots=plots
models=models
null="-"

mkdir -p $plots $models

for x in $null "default" "duluth"; do
    for y in $null "lch" "wup" "res_brown" "lin_brown" "jcn_brown" \
                   "pmi_brown" "all"; do

        if [ $x = $null ] && [ $y = $null ]; then
            continue
        elif [ $x = $null ]; then
            name_transform=wordnet_$y
            name=wordnet_$y
        elif [ $y = $null ]; then
            name_transform=$x
            name=$x
        else
            name_transform=$x+wordnet_$y
            name=$x-wordnet_$y
        fi

        echo "======= $name_transform ======="
        python -u model_train.py --transform $name_transform \
            --alpha 0.1 \
            --epochs 10 \
            --plot $plots/train-$name.png \
            --save $models/$name.pt \
            --transformer_cache cache/transformer-roberta-train \
            --wordnet_cache cache/wordnet-train && \
        python -u model_test.py $models/$name.pt \
            --transformer_cache cache/transformer-roberta-test \
            --wordnet_cache cache/wordnet-test
        echo

    done
done
