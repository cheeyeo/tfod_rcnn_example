# Preprocess LISA dataset into records format accepted by TFOD API
import json
import argparse
import os

import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf

import config.lisa_config as config
from utils.tfannotations import TFAnnotation

def visualize(img_path, xmin, xmax, ymin, ymax):
    fname = img_path.split(os.path.sep)[-1]
    image = cv2.imread(img_path)
    (h, w, _) = image.shape

    startX = int(xmin * w)
    endX = int(xmax * w)
    startY = int(ymin * h)
    endY = int(ymax * h)

    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
    cv2.imwrite("vis/{}".format(fname), image)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true", help="Store images for visualizing")
    args = vars(ap.parse_args())

    with open(config.CLASSES_PATH, "w") as f:
        for k, v in config.CLASSES.items():
            item = ("item {\n"
                      "\tid: " + str(v) + "\n"
                      "\tname: '" + k + "'\n"
                      "}\n")
            f.write(item)

    D = {}
    with open(config.ANNOT_PATH, "r") as f:
        rows = f.read().strip().split("\n")

    for row in rows[1:]:
        row = row.split(",")[0].split(";")
        (img_path, label, startX, startY, endX, endY, _) = row
        (startX, startY) = (float(startX), float(startY))
        (endX, endY) = (float(endX), float(endY))

        if label not in config.CLASSES:
            continue

        # each image may contain mutiple signs
        # hence store image name as key in dict
        imgp = os.path.join(config.BASE_PATH, img_path)
        b = D.get(imgp, [])
        b.append((label, (startX, startY, endX, endY)))
        D[imgp] = b

    # create train/test splits
    train_set, test_set = train_test_split(list(D.keys()), test_size=config.TEST_SIZE, random_state=42)

    datasets = [
        ("train", train_set, config.TRAIN_RECORD),
        ("test", test_set, config.TEST_RECORD)
    ]

    # loop over dataset
    # write to tf record format
    for (dtype, keys, output_path) in datasets:
        print("[INFO] Processing {}".format(dtype))

        writer = tf.io.TFRecordWriter(output_path)

        total = 0
        debugcount = 0
        for k in keys:
            encoded = tf.io.gfile.GFile(k, "rb").read()
            encoded = bytes(encoded)

            pilimg = Image.open(k)
            (w, h) = pilimg.size[:2]

            filename = k.split(os.path.sep)[-1]
            encoding = filename[filename.rfind(".") + 1:]

            tfannot = TFAnnotation()
            tfannot.image = encoded
            tfannot.encoding = encoding
            tfannot.filename = filename
            tfannot.width = w
            tfannot.height = h

            for (label, (startX, startY, endX, endY)) in D[k]:
                # normalize bbox
                xMin = startX / w
                xMax = endX / w
                yMin = startY / h
                yMax = endY / h

                if args["debug"]:
                    if debugcount <= 100:
                        visualize(k, xMin, xMax, yMin, yMax)
                        debugcount += 1

                tfannot.xMins.append(xMin)
                tfannot.xMaxs.append(xMax)
                tfannot.yMins.append(yMin)
                tfannot.yMaxs.append(yMax)
                tfannot.textLabels.append(label.encode("utf8"))
                tfannot.classes.append(config.CLASSES[label])
                tfannot.difficult.append(0)

                total += 1
                features = tf.train.Features(feature=tfannot.build())
                example = tf.train.Example(features=features)

                writer.write(example.SerializeToString())
        writer.close()
        print("[INFO] {} examples saved for {}".format(total, dtype))

