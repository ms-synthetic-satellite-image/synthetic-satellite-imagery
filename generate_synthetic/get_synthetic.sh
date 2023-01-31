#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
DATA_FOLDER="$1"
SPADE_MODEL="$2"
NO_OUTPUT="$3"
GPU_ID="$4"

echo "Generating synthetic tiles"
python inference.py \
--data_folder $DATA_FOLDER \
--spade_model $SPADE_MODEL \
--no_output $NO_OUTPUT \
--gpu_id $GPU_ID 
