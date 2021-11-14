### Faster-RCNN using TFOD

[LISA Traffic signs dataset]: http://cvrr.ucsd.edu/LISA/lisa-traffic-sign-dataset.html

[TensorFlow Object Detection API]: https://github.com/tensorflow/models/tree/master/research/object_detection

[TF2 Model Zoo]: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md 

[TFOD setup using TF 2]: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md

[Faster R-CNN Resnet101 V1 model]: http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet101_v1_800x1333_coco17_gpu-8.tar.gz


[Original blog post on TFOD]: https://cheeyeo.uk/machine-learning/deep-learning/computer-vision/tensorflow/2021/11/03/using-tensorflow-object-detection-api/

[m1l0/tfod tooklit]: https://github.com/m1l0ai/m1l0-tfod


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


### Local Setup and training process

Below describes the steps I took to train the [LISA Traffic signs dataset] using TFOD API.

* Clone the [m1l0/tfod tooklit] and use it as a base working directory.

* Clone the TFOD models zoo into base working directory as **models** and run the following:

```
git clone https://github.com/tensorflow/models.git

cd models/research/

protoc object_detection/protos/*.proto --python_out=.

cp object_detection/packages/tf2/setup.py .

python -m pip install .

# if no errors then it works...
python object_detection/builders/model_builder_tf2_test.py
```

* If there are errors with the test script, resolve them first before moving on to the steps below.

* Download the [LISA Traffic signs dataset] into this working dir as **lisa**. Create the following subdirs: **lisa/records**; **lisa/experiments**; **lisa/experiments/exported_model**; **lisa/experiments/training**


* Run `python build_lisa_records.py` which will output the following files:

  * **lisa/records/training.record**

    The training dataset

  * **lisa/records/testing.record**

    The test dataset

  * **lisa/records/classes.pbtxt**

    Mapping of target class labels to integer values


* Run `train.sh` with the following parameters:

  ```
  ./train.sh models \
  lisa/experiments/training \
  lisa/experiments/exported_model \
  lisa/records \
  "Faster R-CNN ResNet101 V1 800x1333" \
  <num_classes> \
  <min_dim> \
  <max_dim> \
  <num_steps> \
  <batch_size> \
  <num_test_examples>
  ```

  <models> => Directory of tfod models clone
  <model_dir> => Directory where training artifacts stored
  <exported_model_dir> => Directory to where exported model artifacts stored
  <pretrained_model_name> => Pretrained model name e.g. "Faster R-CNN ResNet101 V1 800x1333"
  <num_classes> => Number of labels / categories in config file
  <min_dim> => Min dim in config file
  <max_dim> => Max dim in config file
  <num_steps> => Num of training steps
  <batch_size> => Batch size; must match num of gpus
  <num_of_test_samples> => Number of test samples for evaluation

  Invoking `train.sh` will:

  * Download the required pretrained model as specified via `pretrained_model_name`, extract and save it to the training subdir

  * Sets ENV vars and run `python readconfig.py` which copies the sample pipeline config file from pretrained model dir, and creates a new pipeline config file based on the additional parameters defined above i.e. <num_classes>, <min_dim>, <max_dim>, <num_steps>, <batch_size>, <num_of_test_samples>

  * Starts the training process and logs output to STDOUT, saves model checkpoints to **lisa/experiments/training**

  * Runs evaluation after training completes

  * Saves the final trained model to **lisa/experiments/exported_model**


### Run on AWS

* Before training on AWS, you need to create a TFOD docker image by building the dockerfile in the m1l0/tfod project.

* To train on AWS, a set of terraform scripts are provided in the `terraform folder`. Adjust `terraform/config.tfvars` and then run `make setup` followed by `make apply`

* After the resources are provisioned, run `make runtask` which will create an ECS task and tails the training logs.

* The model artifacts will be saved into the S3 buckets specified in `terraform/config.tfvars`



### Results of initial run

For the purposes of evaluating the Faster-RCNN model on the [LISA Traffic signs dataset], the model was packaged as a docker image and trained on a single p3.2xlarge instance with 1 GPU, 16GB GPU RAM.

The overall training time took approximately 2 hours.

Training config:
* num_steps: 50000
* min_dim: 600
* max_dim: 1024
* num_classes: 3
* batch_size: 1
* optimizer: SGD
* learning rate: 0.01

The rest of the config are kept as it is from the sample config file provided by the pre-trained model.

The SGD optimizer is used with a momentum of **0.9**. 

The learning rate is set to **0.01** with a cosine learning rate decay over the total number of training steps.

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

The overall loss is =~ 0.24 which is high. The mAP@0.5 is 0.731.

We will use the above as a baseline model.


### Further extensions

#### Update the default optimizer

Update the optimizer in the config file to use a lower learning rate without any decay i.e. set the config to manual_learning_rate


TODO