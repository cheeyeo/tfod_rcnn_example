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
```

* Run `train.sh` with the following parameters:

  ```
  ./train.sh models lisa/experiments/training lisa/experiments/exported_model lisa/records faster_rcnn_resnet101_v1_800x1333_coco17_gpu-8 <num_classes> <min_dim> <max_dim> <num_steps> <batch_size> <num_test_examples>
  ```

  <models> => Directory of tfod models clone
  <model_dir> => Directory where training artifacts stored
  <exported_model_dir> => Directory to where exported model artifacts stored
  <pretrained_model_name> => Pretrained model name e.g. "faster_rcnn_resnet101_v1_800x1333_coco17_gpu-8"
  <num_classes> => Number of labels / categories in config file
  <min_dim> => Min dim in config file
  <max_dim> => Max dim in config file
  <num_steps> => Num of training steps
  <batch_size> => Batch size; must match num of gpus
  <num_of_test_samples> => Number of test samples for evaluation


### Results of initial run

Train on single p3.2xlarge instance with 1 GPU, 16GB GPU RAM for 50000 steps

Train with following config:

* num_steps: 50000
* min_dim: 600
* max_dim: 1024
* num_classes: 3
* batch_size: 1

The evaluation results are as follows:
```
2021-11-12T21:46:47 Accumulating evaluation results...
2021-11-12T21:46:48 DONE (t=0.84s).
2021-11-12T21:46:48  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.248
2021-11-12T21:46:48  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.731
2021-11-12T21:46:48  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.101
2021-11-12T21:46:48  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.123
2021-11-12T21:46:48  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.348
2021-11-12T21:46:48  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.654
2021-11-12T21:46:48  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.301
2021-11-12T21:46:48  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.340
2021-11-12T21:46:48  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.364
2021-11-12T21:46:48  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.287
2021-11-12T21:46:48  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.421
2021-11-12T21:46:48  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.692
2021-11-12T21:46:48 INFO:tensorflow:Eval metrics at step 50000
2021-11-12T21:46:48 I1112 21:46:48.684187 139908352481088 model_lib_v2.py:1007] Eval metrics at step 50000
2021-11-12T21:46:48 INFO:tensorflow:  + DetectionBoxes_Precision/mAP: 0.247610
2021-11-12T21:46:48 I1112 21:46:48.692481 139908352481088 model_lib_v2.py:1010]   + DetectionBoxes_Precision/mAP: 0.247610
2021-11-12T21:46:48 INFO:tensorflow:  + DetectionBoxes_Precision/mAP@.50IOU: 0.730677
2021-11-12T21:46:48 I1112 21:46:48.693813 139908352481088 model_lib_v2.py:1010]   + DetectionBoxes_Precision/mAP@.50IOU: 0.730677
2021-11-12T21:46:48 INFO:tensorflow:  + DetectionBoxes_Precision/mAP@.75IOU: 0.100935
2021-11-12T21:46:48 I1112 21:46:48.695106 139908352481088 model_lib_v2.py:1010]   + DetectionBoxes_Precision/mAP@.75IOU: 0.100935
2021-11-12T21:46:48 INFO:tensorflow:  + DetectionBoxes_Precision/mAP (small): 0.123162
2021-11-12T21:46:48 I1112 21:46:48.696399 139908352481088 model_lib_v2.py:1010]   + DetectionBoxes_Precision/mAP (small): 0.123162
2021-11-12T21:46:48 INFO:tensorflow:  + DetectionBoxes_Precision/mAP (medium): 0.348016
2021-11-12T21:46:48 I1112 21:46:48.697730 139908352481088 model_lib_v2.py:1010]   + DetectionBoxes_Precision/mAP (medium): 0.348016
2021-11-12T21:46:48 INFO:tensorflow:  + DetectionBoxes_Precision/mAP (large): 0.654373
2021-11-12T21:46:48 I1112 21:46:48.699023 139908352481088 model_lib_v2.py:1010]   + DetectionBoxes_Precision/mAP (large): 0.654373
2021-11-12T21:46:48 INFO:tensorflow:  + DetectionBoxes_Recall/AR@1: 0.300572
2021-11-12T21:46:48 I1112 21:46:48.700293 139908352481088 model_lib_v2.py:1010]   + DetectionBoxes_Recall/AR@1: 0.300572
2021-11-12T21:46:48 INFO:tensorflow:  + DetectionBoxes_Recall/AR@10: 0.339940
2021-11-12T21:46:48 I1112 21:46:48.701619 139908352481088 model_lib_v2.py:1010]   + DetectionBoxes_Recall/AR@10: 0.339940
2021-11-12T21:46:48 INFO:tensorflow:  + DetectionBoxes_Recall/AR@100: 0.363816
2021-11-12T21:46:48 I1112 21:46:48.702923 139908352481088 model_lib_v2.py:1010]   + DetectionBoxes_Recall/AR@100: 0.363816
2021-11-12T21:46:48 INFO:tensorflow:  + DetectionBoxes_Recall/AR@100 (small): 0.286649
2021-11-12T21:46:48 I1112 21:46:48.704222 139908352481088 model_lib_v2.py:1010]   + DetectionBoxes_Recall/AR@100 (small): 0.286649
2021-11-12T21:46:48 INFO:tensorflow:  + DetectionBoxes_Recall/AR@100 (medium): 0.420546
2021-11-12T21:46:48 I1112 21:46:48.705553 139908352481088 model_lib_v2.py:1010]   + DetectionBoxes_Recall/AR@100 (medium): 0.420546
2021-11-12T21:46:48 INFO:tensorflow:  + DetectionBoxes_Recall/AR@100 (large): 0.691667
2021-11-12T21:46:48 I1112 21:46:48.706911 139908352481088 model_lib_v2.py:1010]   + DetectionBoxes_Recall/AR@100 (large): 0.691667
2021-11-12T21:46:48 INFO:tensorflow:  + Loss/RPNLoss/localization_loss: 0.003441
2021-11-12T21:46:48 I1112 21:46:48.707967 139908352481088 model_lib_v2.py:1010]   + Loss/RPNLoss/localization_loss: 0.003441
2021-11-12T21:46:48 INFO:tensorflow:  + Loss/RPNLoss/objectness_loss: 0.118597
2021-11-12T21:46:48 I1112 21:46:48.709069 139908352481088 model_lib_v2.py:1010]   + Loss/RPNLoss/objectness_loss: 0.118597
2021-11-12T21:46:48 INFO:tensorflow:  + Loss/BoxClassifierLoss/localization_loss: 0.064199
2021-11-12T21:46:48 I1112 21:46:48.710147 139908352481088 model_lib_v2.py:1010]   + Loss/BoxClassifierLoss/localization_loss: 0.064199
2021-11-12T21:46:48 INFO:tensorflow:  + Loss/BoxClassifierLoss/classification_loss: 0.058869
2021-11-12T21:46:48 I1112 21:46:48.711216 139908352481088 model_lib_v2.py:1010]   + Loss/BoxClassifierLoss/classification_loss: 0.058869
2021-11-12T21:46:48 INFO:tensorflow:  + Loss/regularization_loss: 0.000000
2021-11-12T21:46:48 I1112 21:46:48.712285 139908352481088 model_lib_v2.py:1010]   + Loss/regularization_loss: 0.000000
2021-11-12T21:46:48 INFO:tensorflow:  + Loss/total_loss: 0.245106
2021-11-12T21:46:48 I1112 21:46:48.713370 139908352481088 model_lib_v2.py:1010]   + Loss/total_loss: 0.245106
```






### Running with docker

* Create the docker image using the provided dockerfile

* Create a `.env` file with following and source it:

  ```
  AWS_PROFILE: Name of aws_profile (optional)

  AWS_ROOT: Path of aws credentials if running locally (optional)
  
  S3_DATA: Path to where training data stored (optional)
  ```

* Run `make local-run` which will start training process using local paths defined in .env or `make ecs-run` which uses s3 dataset


### TODO:

* Automate process of training model in cloud as it uses too much GPU resources locally
