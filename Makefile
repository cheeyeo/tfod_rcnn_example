.PHONY: train dockerrun export

dockerrun:
	docker run --gpus all --rm -v "/media/chee/DISK D/DL4CV/object_detection/tfod_example/lisa/records":/opt/tfod/records -v "/media/chee/DISK D/DL4CV/object_detection/tfod_example/lisa/experiments/training/faster_rcnn_lisa_docker.config":/opt/tfod/experiments/training/faster_rcnn_lisa_docker.config m1l0/tfod:latest

train:
	python models/research/object_detection/model_main_tf2.py \
	--pipeline_config_path="${PIPELINE_CONFIG_PATH}" \
	--model_dir="${MODEL_DIR}" \
	--num_train_steps=50000 \
	--sample_1_of_n_eval_examples=1 \
	--alsologtostderr

export:
	python models/research/object_detection/export_inference_graph.py \
	--input_type image_tensor \
	--pipeline_config_path "${PIPELINE_CONFIG_PATH}" \
	--trained_checkpoint_prefix "${MODEL_DIR}/model.ckpt-50000" \
	--output "${EXPORTED_DIR}"