### Faster-RCNN using TFOD

[LISA Traffic signs dataset]: http://cvrr.ucsd.edu/LISA/lisa-traffic-sign-dataset.html

[TensorFlow Object Detection API]: https://github.com/tensorflow/models/tree/master/research/object_detection

[TF2 Model Zoo]: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md 

[TFOD setup using TF 2]: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md

[Faster R-CNN Resnet101 V1 model]: http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet101_v1_800x1333_coco17_gpu-8.tar.gz


[Original blog post on TFOD]: https://cheeyeo.uk/machine-learning/deep-learning/computer-vision/tensorflow/2021/11/03/using-tensorflow-object-detection-api/


Apply fine-tuning to a trained Faster-RCNN object detection model via the TFOD API.

[Original blog post on TFOD] with more detailed explanations.


### TFOD API

[TensorFlow Object Detection API] consist of pre-trained object detectors from which we can apply transfer learning to allow it to learn it from custom datasets.

The Faster-RCNN architecture is complex, comprising of many components which if being manually implemented, would defintely result in hard to track issues and errors. Using a well-tested, pre-built, open-source, robust framework is the best approach for quickly iterating on testing object detection models on custom datasets.

The Faster-RCNN describes the architecture. The base network is a pre-trained model. For this example we chose the ResNet 101 model. The pre-trained model weights are downloaded and used during the training process through fine-tuning

Notes on the pre-trained models:

* Only the saved model and weights are downloaded.

* The models are trained on the COCO 2017 dataset

* The models are trained on different hardware, e.g. GPU or TPU. Ensure that you select the right model for your specific architecture else it will cause hard to debug errors during training

We use the [Faster R-CNN Resnet101 V1 model] for this example.


### Setup

* Download the [LISA Traffic signs dataset] into this working dir as 'lisa'. Create a subdir of 'records' and 'experiments'

* Run `python build_lisa_records.py` which will output the training/test records and a labels mapping file to 'records' subdir. We need this during training phase.

* Clone the TFOD models zoo into local working dir of this repo and run the following:

```
git clone https://github.com/tensorflow/models.git

cd models/research/

protoc object_detection/protos/*.proto --python_out=.

cp object_detection/packages/tf2/setup.py .

python -m pip install .

# if no errors then it works...
python object_detection/builders/model_builder_tf2_test.py
```

* If there are errors with the running of the test script, resolve them first before moving on to the next step below.

* Create a training directory for the model's checkpoints files during training and to load the pre-trained model's weights. 

```
mkdir -p lisa/experiments/training

curl -L -o faster_rcnn_resnet101_v1_800x1333_coco17_gpu-8.tar.gz http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet101_v1_800x1333_coco17_gpu-8.tar.gz


tar -zxvf faster_rcnn_resnet101_v1_800x1333_coco17_gpu-8.tar.gz

# check that it has a checkpoint dir, saved_model dir, pipeline.config file
ls -al faster_rcnn_resnet101_v1_800x1333_coco17_gpu-8

# cp the pipeline.config file into lisa/experiments/training, rename it
```

* Update the copied pipeline.config file as follows:
```
model {
  faster_rcnn {
    num_classes: 3

   }
}
...

# set batch size to 1 but can be higher if you have compute
# set num steps to 50000; min is 20000
# set fine_tune_checkpoint to directory of 'checkpoint' in the downloaded model's weights; only use the prefix
train_config: {
  batch_size: 1
  num_steps: 50000

  ...
  fine_tune_checkpoint_version: V2
  fine_tune_checkpoint: "<location of model weights>/checkpoint/ckpt-0"
  from_detection_checkpoint: true
  fine_tune_checkpoint_type: "detection"
}

...


train_input_reader: {
  label_map_path: "<location of classes.pbtxt>"
  tf_record_input_reader {
    input_path: "<location of training.record>"
  }
}


...

# num_examples to match actual num of samples in test set
eval_config: {
  metrics_set: "coco_detection_metrics"
  num_examples: 955
}

eval_input_reader: {
  label_map_path: "<location of classes.pbtxt>"
  shuffle: false
  num_epochs: 1
  tf_record_input_reader {
    input_path: "<location of testing.record>"
  }
}

```

* Create a `.env` file with following and source it:

  ```
  AWS_PROFILE: Name of aws_profile (optional)

  AWS_ROOT: Path of aws credentials if running locally (optional)
  
  S3_DATA: Path to where training data stored (optional)
  
  LOCAL_DATA_PATH: Path to where training data held locally
  
  LOCAL_CONFIG_PATH: Path to training config file locally
  ```


### Running with docker

* Create the docker image using the provided dockerfile

* Run `make local-run` which will start training process using local paths defined in .env or `make ecs-run` which uses s3 dataset


### Running manually

* Change back to current working dir of this project and run `source setup.sh <tfod_dir> <model_dir> <model_export_dir> <config_file_name>` where:
  ```
	**tfod_dir**: Cloned models dir
	**model_dir**: Dir where model's weights being extracted to
  **model_export_dir**: Dir to save trained model's weights
	**config_file_name**: Name of config file within the model_dir.
  ```

* Run `make train` or `make export`


### TODO:

* Automate process of training model in cloud as it uses too much GPU resources locally
