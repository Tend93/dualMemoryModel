#!/bin/bash

cd eval
python eval.py ../data/reference_videos_Youtube.json $1
cd ../
