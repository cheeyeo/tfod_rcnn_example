import argparse
import os
import json

from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format


def _update_optimizer_with_manual_step_learning_rate(
    optimizer, initial_learning_rate, learning_rate_scaling):
    """Adds a learning rate schedule."""
    manual_lr = optimizer.learning_rate.manual_step_learning_rate
    manual_lr.initial_learning_rate = initial_learning_rate
    for i in range(3):
        schedule = manual_lr.schedule.add()
        schedule.step = int(i * 200000 * 4.5)
        schedule.learning_rate = initial_learning_rate * learning_rate_scaling**i


def _update_image_resizer(model_config, config):
    """Updates the model's image resizer"""
    architecture = model_config.WhichOneof("model")
    print(architecture)
    print(config)
    if architecture == "faster_rcnn":
        if "keep_aspect_ratio_resizer" in config.keys():
            for k, v in config["keep_aspect_ratio_resizer"].items():
                setattr(model_config.faster_rcnn.image_resizer.keep_aspect_ratio_resizer, k, v)


"""
NOTE:

For most config we can specify the full path in the config file e.g. model.faster_rcnn.image_resizer.keep_aspect_ratio_resizer.min_dimension
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--override", type=str, help="Path to override model config")

    args = vars(ap.parse_args())
    print(args)

    # parse model overrides
    with open(args["override"], "r") as f:
        model_override = json.loads(f.read())

    config_file = os.path.join(os.getcwd(), "lisa", "experiments", "training", "faster_rcnn_resnet101_v1_800x1333_coco17_gpu-8", "pipeline.config")
    orig_configs = config_util.get_configs_from_pipeline_file(config_file)
    # print(orig_configs)
    # print(orig_configs["model"].faster_rcnn)

    # Check if learning rate is set
    if "optimizer" in model_override.keys():
        optimizer_opt = model_override.pop("optimizer")
        print(optimizer_opt)
        optimizer = getattr(orig_configs["train_config"].optimizer, "momentum_optimizer")
        print(optimizer)

        if optimizer_opt["type"] == "manual_step":
            _update_optimizer_with_manual_step_learning_rate(optimizer, optimizer_opt["initial_learning_rate"], 0.1)

    if "image_resizer" in model_override.keys():
        resize_opts = model_override.pop("image_resizer")
        # print(resize_opts)
        # print(orig_configs["model"])
        # print(dir(orig_configs["model"]))
        _update_image_resizer(orig_configs["model"], resize_opts)



    updated_configs = config_util.merge_external_params_with_configs(orig_configs, kwargs_dict=model_override)


    print("Updated config \n{}".format(updated_configs))

    # configs = config_util.create_pipeline_proto_from_configs(orig_configs)
    # # save_pipeline_config takes a dir
    # saved_dir = os.path.join("lisa", "experiments", "training", "testconfig")
    # config_util.save_pipeline_config(configs, saved_dir)
