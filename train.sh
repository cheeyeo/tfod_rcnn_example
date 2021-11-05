#!/bin/bash

set -e
#set -x
set -o pipefail

# usage: source ./train.sh models lisa/experiments/training lisa/experiments/exported_model faster_rcnn_lisa.config <records path>

echo "Setting up paths for TFOD..."
currentdir=$(pwd)
tfod_dir=${currentdir}/${1}
model_dir=${currentdir}/${2}
exported_dir=${currentdir}/${3}
pipeline_config=${model_dir}/${4}
training_data_path=${5}

echo "Current dir is: ${currentdir}"
echo "TFOD Models dir is: ${tfod_dir}"
echo "Model dir: ${model_dir}"
echo "Exported Dir: ${exported_dir}"
echo "Pipeline config filename: ${pipeline_config}"
echo "Train data: ${training_data_path}"

export PYTHONPATH=${PYTHONPATH}:"${tfod_dir}/research":"${tfod_dir}/research/slim"
echo "Updated PYTHONPATH: ${PYTHONPATH}"

export PIPELINE_CONFIG_PATH="${pipeline_config}"
export MODEL_DIR="${model_dir}"
export EXPORTED_DIR="${exported_dir}"


echo "Getting training data..."
if [[ $training_data_path == *"s3"* ]]; then
	echo "S3 FOUND!"

	touch ${currentdir}/records/classes.pbtxt
	touch ${currentdir}/records/training.record
	touch ${currentdir}/records/testing.record

	aws s3 cp --cli-read-timeout 0 ${training_data_path}/classes.pbtxt - > ${currentdir}/records/classes.pbtxt

	aws s3 cp --cli-read-timeout 0 ${training_data_path}/training.record - > ${currentdir}/records/training.record

	aws s3 cp --cli-read-timeout 0 ${training_data_path}/testing.record - > ${currentdir}/records/testing.record
fi

echo "Starting training process..."
python3 models/research/object_detection/model_main_tf2.py \
	--pipeline_config_path="${PIPELINE_CONFIG_PATH}" \
	--model_dir="${MODEL_DIR}" \
	--num_train_steps=50000 \
	--sample_1_of_n_eval_examples=1 \
	--alsologtostderr

echo "Exporting model..."
python3 models/research/object_detection/export_inference_graph.py \
	--input_type image_tensor \
	--pipeline_config_path "${PIPELINE_CONFIG_PATH}" \
	--trained_checkpoint_prefix "${MODEL_DIR}/model.ckpt-50000" \
	--output "${EXPORTED_DIR}"