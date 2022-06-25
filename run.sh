#!/bin/bash

gpus=(1 2 4 5 6)

for i_run in {0..4}; do
    CUDA_VISIBLE_DEVICES=${gpus[$i_run]} python exp/tcgan_.py --i_run=$i_run &
done
