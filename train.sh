#!/bin/bash

flaginfo="double_state_attention_just_test"
flag="-flag "$flaginfo
datainput=" -label_file /home/yzw/dataset/Youtube/feature/new_lv/labels_1th.npy -input_h5 /home/yzw/dataset/Youtube/new_feat/googlenet-pl5-10f-new.h5 -input_json /home/yzw/dataset/Youtube/feature/new_lv/info_1th.json "
topic=" -input_h5_local /home/yzw/dataset/Youtube/new_feat/googlenet-in5b-10f-new.h5 "

params=" -gpuid 0 -learningRate 2e-4 -dropout 0.5 -grad_clip 10 -learning_rate_decay_every 20 -add_supervision 0 "
size=" -batchsize 64 -hiddensize 512 -eval_every 100 "
cmd=" th train_dual_memory_model.lua -checkpoint_dir checkpoints -max_epochs 80 "

a=$(date +%y%m%d-%H%M)

log="logs/"$a$flaginfo".log"

$cmd $datainput $topic $params $size $flag | tee $log
echo "$cmd $datainput $topic $params $size $flag" >> $log
wait



