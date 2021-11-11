.PHONY: train local-run ecs-run setup apply teardown

local-run:
	docker run --gpus all --rm -v "${LOCAL_DATA_PATH}":/opt/tfod/records m1l0/tfod:latest "models" "experiments/training" "experiments/exported_model" "records" "faster_rcnn_resnet101_v1_800x1333_coco17_gpu-8" 3 600 1024 50000 1 955

ecs-run:
	docker run --gpus all --rm -e AWS_PROFILE=${AWS_PROFILE} -v "${AWS_ROOT}":"/root/.aws:ro" m1l0/tfod:latest "models" "experiments/training" "experiments/exported_model" ${S3_DATA} "faster_rcnn_resnet101_v1_800x1333_coco17_gpu-8" 3 600 1024 50000 1 955

train:
	python models/research/object_detection/model_main_tf2.py \
	--pipeline_config_path="${PIPELINE_CONFIG_PATH}" \
	--model_dir="${MODEL_DIR}" \
	--num_train_steps=50000 \
	--sample_1_of_n_eval_examples=1 \
	--alsologtostderr

setup:
	terraform -chdir=terraform init
	terraform -chdir=terraform plan -out=myplan -var-file=config.tfvars

apply:
	terraform -chdir=terraform apply myplan

teardown:
	terraform -chdir=terraform destroy -var-file=config.tfvars

runtask:
	terraform -chdir=terraform output -json > configs.json
	./runtask.sh configs.json