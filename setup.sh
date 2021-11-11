#!/bin/bash

# usage: source ./setup.sh models lisa/experiments/training lisa/experiments/exported_model lisa/records faster_rcnn_resnet101_v1_800x1333_coco17_gpu-8 <num_classes> <min_dim> <max_dim> <num_steps> <batch_size> <num_examples>

echo "Setting up paths for TFOD..."
currentdir=$(pwd)
tfod_dir=${currentdir}/${1}
model_dir=${currentdir}/${2}
exported_dir=${currentdir}/${3}
records_dir=${currentdir}/${4}
pretrained_model_dir=${model_dir}/${5}
# generated config file path
pipeline_config="${model_dir}/pipeline.config"

num_classes=${6}
min_dim=${7}
max_dim=${8}
num_steps=${9}
batch_size=${10}
num_examples=${11}

echo "Current dir is: ${currentdir}"
echo "TFOD Models dir is: ${tfod_dir}"
echo "Model dir: ${model_dir}"
echo "Exported Dir: ${exported_dir}"
echo "Training Data Dir: ${records_dir}"
echo "Pipeline config filename: ${pipeline_config}"

export PYTHONPATH=${PYTHONPATH}:"${tfod_dir}/research":"${tfod_dir}/research/slim"
echo "Updated PYTHONPATH: ${PYTHONPATH}"

export PIPELINE_CONFIG_PATH="${pipeline_config}"
export MODEL_DIR="${model_dir}"
export EXPORTED_DIR="${exported_dir}"
export RECORDS_DIR="${records_dir}"
export PRETRAINED_MODEL_DIR="${pretrained_model_dir}"

# echo "Generating config file for training..."
# python readconfig.py --num_classes=${num_classes} \
#                      --min_dim=${min_dim} \
# 		                 --max_dim=${max_dim} \
# 		                 --num_steps=${num_steps} \
# 		                 --batch_size=${batch_size} \
# 		                 --num_examples=${num_examples}