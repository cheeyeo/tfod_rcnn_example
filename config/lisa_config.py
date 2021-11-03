import os


# base path for images
BASE_PATH = "lisa"

ANNOT_PATH = os.path.join(BASE_PATH, "allAnnotations.csv")

TRAIN_RECORD = os.path.join(BASE_PATH, "records", "training.record")

TEST_RECORD = os.path.join(BASE_PATH, "records", "testing.record")

CLASSES_PATH = os.path.join(BASE_PATH, "records", "classes.pbtxt")

TEST_SIZE = 0.25

CLASSES = {"pedestrianCrossing": 1, "signalAhead": 2, "stop": 3}