# https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/auto_examples/plot_object_detection_checkpoint.html#sphx-glr-auto-examples-plot-object-detection-checkpoint-py

# Usage
# python predict.py --model lisa/experiments/exported_model \
# --labels lisa/records/classes.pbtxt \
# --num_classes 3 --image lisa/vid0/frameAnnotations-vid_cmp2.avi_annotations/pedestrian_1323804463.avi_image1.png

import argparse
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress TensorFlow logging (1)
import time

import cv2
import imutils
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR') # Suppress TensorFlow logging (2)


from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils


def load_saved_model(model_path):
    """
    Loads the saved model
    """
    model = tf.saved_model.load(os.path.join(model_path, "saved_model"))
    return model


def load_image(img_path):
    """
    Loads image into np array
    """
    img = cv2.imread(img_path)
    (h, w) = img.shape[:2]

    if w > h and w > 1000:
        img = imutils.resize(img, width=1000)
    elif h > w and h > 1000:
        img = imutils.resize(img, height=1000)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return (h, w), np.asarray(img)


if __name__ == "__main__":
    logger = logging.getLogger()

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="Path for exported model")
    ap.add_argument("--labels", type=str, required=True, help="Path to Labels File")
    ap.add_argument("--image", type=str, help="Path to input image")
    ap.add_argument("--num_classes", type=int, help="Num of class labels")
    ap.add_argument("--min_confidence", type=float, default=0.5, help="Min prob to filter weak detections...")
    ap.add_argument("--output_file", type=str, default="result.png", help="Path to output file")

    args = vars(ap.parse_args())

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # set of colors for class labels
    COLORS = np.random.uniform(0, 255, size=(args["num_classes"], 3))

    print("Loading model...")
    start_time = time.time()

    detection_model = load_saved_model(args["model"])

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Model loading took {} secs".format(elapsed_time))

    category_idx = label_map_util.create_category_index_from_labelmap(args["labels"], use_display_name=True)

    (h, w), image_np = load_image(args["image"])
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = np.expand_dims(input_tensor, axis=0)
    image_copy = image_np.copy()

    detections = detection_model(input_tensor)
    num_detections = int(detections.pop("num_detections"))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections["num_detections"] = num_detections
    detections["detection_classes"] = detections["detection_classes"].astype(np.int64)

    boxes = detections["detection_boxes"]
    scores = detections["detection_scores"]
    labels = detections["detection_classes"]

    for (box, label, score) in zip(boxes, labels, scores):
        if score < args["min_confidence"]:
            continue

        (startY, startX, endY, endX) = box
        startX = int(startX * w)
        startY = int(startY * h)
        endX = int(endX * w)
        endY = int(endY * h)

        label = category_idx[label]
        idx = int(label["id"] - 1)
        label = "{}: {:.2f}".format(label["name"], score)
        print("[INFO] Prediction: {}".format(label))

        cv2.rectangle(image_copy, (startX, startY), (endX, endY), COLORS[idx], 2)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(image_copy, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLORS[idx], 1)

    # cv2.imshow("Output", image_copy)
    # cv2.waitKey(0)
    cv2.imwrite(args["output_file"], image_copy)