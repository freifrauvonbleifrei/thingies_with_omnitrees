#!/usr/bin/env bash

numbers_boxes="16 32 64 128 256 512 1024 2048"
# find all the integer prefixes of svg files
prefix_numbers=$(ls -1 | grep -Eo '^[0-9]+' | sort -n | uniq)
echo $prefix_numbers

#iterate
for number in $prefix_numbers; do
    expressions="${number}_octree_%s_*eval.svg ${number}_omnitree_1_%s_*eval.svg ${number}_omnitree_2_%s_*eval.svg ${number}_omnitree_3_%s_*eval.svg"
    echo $expressions
    for expression in $expressions; do
        INPUTS=" "
        for number in $numbers_boxes; do
            expression_with_number=$(printf -- "$expression " $number)
            INPUTS=$INPUTS$expression_with_number
        done
        echo $INPUTS
        for i in $INPUTS; do
            # install with sudo npm install svgexport -g
            svgexport $i $i.png
            #  1024:1024
        done

        convert $(for i in $INPUTS; do printf -- "-delay 50 %s " $i.png; done) ${expression::-13}.gif

        for i in $INPUTS; do
            rm $i.png
        done
    done
done
