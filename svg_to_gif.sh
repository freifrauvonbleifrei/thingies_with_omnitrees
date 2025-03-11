#!/usr/bin/env bash

numbers_boxes="16 32 64 128 256 512 1024 2048"
expression="100349_omnitree_3_%s_s64_eval.svg"
INPUTS=" "
for number in $numbers_boxes ; do
    INPUTS=$INPUTS$(printf -- " $expression " $number)
done

echo $INPUTS

for i in $INPUTS ; do
    # install with sudo npm install svgexport -g
    svgexport $i $i.png
    #  1024:1024
done

convert $(for i in $INPUTS; do printf -- "-delay 50 %s " $i.png ; done; ) 100349_omnitree_3.gif

for i in $INPUTS ; do
    rm $i.png
done