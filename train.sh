#!/bin/bash

set -ex
set -o pipefail

# usage: source ./train.sh models lisa/experiments/training lisa/experiments/exported_model lisa/records faster_rcnn_resnet101_v1_800x1333_coco17_gpu-8 <num_classes> <min_dim> <max_dim> <num_steps> <batch_size> <num_examples>

echo "Setting up paths for TFOD..."
currentdir=$(pwd)
tfod_dir=${currentdir}/${1}
model_dir=${currentdir}/${2}
exported_dir=${currentdir}/${3}

if [[ ${4} == *"s3"* ]]; then
	records_dir=${4}
else
  records_dir=${currentdir}/${4}
fi


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

echo "Getting training data..."
if [[ $records_dir == *"s3"* ]]; then
	echo "S3 FOUND!"

	RECORDS_PATH=/opt/tfod/records

	mkdir -p /tmp/records
	mkfifo /tmp/records/classes.pbtxt
	mkfifo /tmp/records/training.record
	mkfifo /tmp/records/testing.record

	aws s3 cp --cli-read-timeout 0 ${records_dir}/classes.pbtxt - > /tmp/records/classes.pbtxt &

	aws s3 cp --cli-read-timeout 0 ${records_dir}/training.record - > /tmp/records/training.record &

	aws s3 cp --cli-read-timeout 0 ${records_dir}/testing.record - > /tmp/records/testing.record &

	python3 readfifo.py --input_dir /tmp/records --output "${RECORDS_PATH}"

	echo "Waiting for named pipes to close..."
	sleep 5
	echo "Done"

	echo "Removing named pipes"
	rm -rf /tmp/records

	export RECORDS_DIR="${RECORDS_PATH}"
fi

echo "Generating config file for training..."
python3 readconfig.py --num_classes=${num_classes} \
                      --min_dim=${min_dim} \
		                  --max_dim=${max_dim} \
		                  --num_steps=${num_steps} \
		                  --batch_size=${batch_size} \
		                  --num_examples=${num_examples}

echo "Starting training process..."
python3 models/research/object_detection/model_main_tf2.py \
	--pipeline_config_path="${PIPELINE_CONFIG_PATH}" \
	--model_dir="${MODEL_DIR}" \
	--num_train_steps=${num_steps} \
	--sample_1_of_n_eval_examples=1 \
	--alsologtostderr

echo "Exporting model..."
python3 models/research/object_detection/export_inference_graph.py \
	--input_type image_tensor \
	--pipeline_config_path "${PIPELINE_CONFIG_PATH}" \
	--trained_checkpoint_prefix "${MODEL_DIR}/model.ckpt-${num_steps}" \
	--output "${EXPORTED_DIR}"