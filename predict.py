# https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/auto_examples/plot_object_detection_checkpoint.html#sphx-glr-auto-examples-plot-object-detection-checkpoint-py

import argparse

import cv2
import imutils
import numpy as np
from object_detection.utils import label_map_util
import tensorflow as tf


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path for exported model")
    ap.add_argument("--labels", required=True, help="Path to Labels File")
    ap.add_argument("--image", help="Path to input image")
    ap.add_argument("--num_classes", type=int, help="Num of class labels")
    ap.add_argument("--min_confidence", type=float, default=0.5, help="Min prob to filter weak detections...")

    args = vars(ap.parse_args())

    # set of colors for class labels
    COLORS = np.random.uniform(0, 255, size=(args["num_classes"], 3))

    model = tf.Graph()

    with model.as_default():
        graph_def = tf.GraphDef()

        with tf.gfile.io.GFile(args["model"], "rb") as f:
            serialized_grpah = f.read()
            graph_def.ParseFromString(serialized_grpah)
            tf.import_graph_def(graph_def, name="")

    print(model)